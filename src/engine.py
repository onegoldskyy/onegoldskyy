"""가상 묶음 생성, 성과지표 계산, 사전 고정 판단 기준 적용.

전략(spec)은 dict 로 표현한다.
  A: {"code": "A0"|"A1"|"A2"|"A3"|"A1+A2", "dec_th": float|None, "stab_th": float|None}
  B: {"code": "B0"|"B1"|"B2"|"B3"|"B4", "decade": str|None}
  C: "C0"|"C1"|"C2"|"C1+C2"
  N: int
  E: "E1"|"E2"|"E3"|"E4"
선정 단계에서는 sel(구성 시점 정보)만 쓰고, 성과 계산에서만 out(평가기간 결과)을 쓴다.
"""
import json
import math
import zlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import SEED, N_BUNDLES, PHASES
from features import build_phase, monthly_eval_matrix, monthly_hist_matrix

BROADCAST_HEAVY = 0.5     # C1: 방송 비중 50% 이상
AGE_MIN = 7.0             # A1
E4_LB, E4_UB = 0.01, 0.15  # E4 곡당 비중 제약
COVERAGE_MIN = 0.8        # 총수익률 집계 최소 가격 커버리지


# ---------------------------------------------------------------- 표기
def spec_name(s: dict) -> str:
    a = s["A"]
    an = a["code"]
    if a.get("dec_th") is not None:
        an = an.replace("A2", f"A2(감소≤{a['dec_th']:.0%})")
    if a.get("stab_th") is not None:
        an = an.replace("A3", f"A3(|LTM/LTY−1|≤{a['stab_th']})")
    b = s["B"]["code"] + (f"[{s['B']['decade']}]" if s["B"].get("decade") else "")
    return f"{an} / {b} / {s['C']} / N={s['N']} / {s['E']}"


def complexity(s: dict) -> int:
    """규칙 수 (판단 기준 3: 단순한 쪽 우선). 동일 금액(E1)이 가장 단순."""
    a = {"A0": 0, "A1": 1, "A2": 1, "A3": 1, "A1+A2": 2}[s["A"]["code"]]
    b = 0 if s["B"]["code"] == "B0" else 1
    c = {"C0": 0, "C1": 1, "C2": 1, "C1+C2": 2}[s["C"]]
    e = {"E1": 0, "E2": 1, "E3": 1, "E4": 2}[s["E"]]
    return a + b + c + e


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode("utf-8"))])


# ---------------------------------------------------------------- 후보군
def apply_A(sel: pd.DataFrame, A: dict) -> pd.Index:
    m = sel["eligible"].copy()
    code = A["code"]
    if code in ("A1", "A1+A2"):
        m &= sel["age"] >= AGE_MIN
    if code in ("A2", "A1+A2"):
        m &= sel["hist_decline"] <= A["dec_th"] + 1e-12
    if code == "A3":
        m &= sel["stab"] <= A["stab_th"] + 1e-12
    return sel.index[m]


def build_pool(sel: pd.DataFrame, elig: pd.Index, B: dict, N: int):
    """B 규칙에 따른 추출 후보군과 (B4의 경우) 층 정보."""
    code = B["code"]
    s = sel.loc[elig]
    if code == "B0" or code == "B4":
        return s.index
    if code == "B1":
        return s.index[s["decade"] == B["decade"]]
    if code == "B2":
        score = (s["hist_yield"].rank(ascending=False) + s["hist_decline"].rank(ascending=True)) / 2
        return score.sort_values(kind="mergesort").index[:3 * N]
    if code == "B3":
        return s["stab"].fillna(np.inf).sort_values(kind="mergesort").index[:3 * N]
    raise ValueError(code)


def _quotas(strata: pd.Series, N: int) -> dict:
    """후보군 주요 수익원 구성비대로 N을 배분 (최대잉여법)."""
    p = strata.value_counts(normalize=True)
    raw = p * N
    q = np.floor(raw).astype(int)
    rem = N - q.sum()
    for k in (raw - q).sort_values(ascending=False, kind="mergesort").index[:rem]:
        q[k] += 1
    return q.to_dict()


def sample_bundles(sel: pd.DataFrame, pool: pd.Index, spec: dict, tag: str, n_bundles=N_BUNDLES):
    """조건을 지키며 무작위 순서로 곡을 채운다. (n_bundles × N) id 배열과 실패 수 반환."""
    N, C, B = spec["N"], spec["C"], spec["B"]["code"]
    rng = _rng(tag)
    ids = np.asarray(pool)
    heavy = (sel.loc[ids, "broadcast_share"] >= BROADCAST_HEAVY).to_numpy()
    artist = sel.loc[ids, "singer"].to_numpy()
    strata = sel.loc[ids, "main_source"].to_numpy()
    heavy_cap = math.ceil(N / 10)
    use_c1 = C in ("C1", "C1+C2")
    use_c2 = C in ("C2", "C1+C2")
    quotas = _quotas(pd.Series(strata), N) if B == "B4" else None
    out = np.empty((n_bundles, N), dtype=ids.dtype)
    fails = 0
    for b in range(n_bundles):
        perm = rng.permutation(len(ids))
        if not (use_c1 or use_c2 or quotas):
            out[b] = ids[perm[:N]]
            continue
        chosen, n_heavy, arts = [], 0, set()
        filled = {k: 0 for k in quotas} if quotas else None

        def ok(j, check_quota):
            if use_c1 and heavy[j] and n_heavy >= heavy_cap:
                return False
            if use_c2 and artist[j] in arts:
                return False
            if check_quota and filled[strata[j]] >= quotas[strata[j]]:
                return False
            return True

        for pass_quota in ([True, False] if quotas else [False]):
            for j in perm:
                if len(chosen) == N:
                    break
                if j in chosen or not ok(j, pass_quota):
                    continue
                chosen.append(j)
                n_heavy += heavy[j]
                arts.add(artist[j])
                if filled is not None:
                    filled[strata[j]] += 1
        if len(chosen) < N:
            fails += 1
            chosen += [j for j in perm if j not in chosen][: N - len(chosen)]
        out[b] = ids[np.array(chosen)]
    return out, fails


# ---------------------------------------------------------------- 비중
def _ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf(2004) 축소 공분산 (목표: 평균분산 × 단위행렬)."""
    T, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = ((S - F) ** 2).sum() / p
    b2 = sum(((np.outer(x, x) - S) ** 2).sum() for x in Xc) / p / T ** 2
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S


def _minvar(cov: np.ndarray, lb: float, ub: float) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.full(n, 1 / n)
    res = minimize(lambda w: w @ cov @ w, x0, jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(lb, ub)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x if res.success else x0, lb, ub)
    return w / w.sum()


def weights(sel, bundles: np.ndarray, E: str, hist_yield_m: pd.DataFrame | None = None) -> np.ndarray:
    nb, N = bundles.shape
    if E == "E1":
        return np.full((nb, N), 1 / N)
    if E == "E2":   # 기대 지급(LTM 기준)이 같도록: w ∝ 가격/LTM
        v = (sel["base_price"] / sel["ltm"].replace(0, np.nan)).reindex(bundles.ravel()).to_numpy().reshape(nb, N)
        v = np.where(np.isfinite(v), v, np.nanmax(v[np.isfinite(v)]))
    elif E == "E3":  # 곡 전체 지급 규모 비례
        v = sel["ltm_total"].reindex(bundles.ravel()).to_numpy().reshape(nb, N).clip(min=0)
    elif E == "E4":
        lb, ub = E4_LB, max(E4_UB, 1 / N)  # N<7이면 상한 15%로는 합 1이 불가능 → 1/N로 완화
        Y = hist_yield_m
        out = np.empty((nb, N))
        for b in range(nb):
            X = Y.loc[bundles[b]].T
            X = X.fillna(X.mean()).to_numpy()
            out[b] = _minvar(_ledoit_wolf(X), lb, ub)
        return out
    else:
        raise ValueError(E)
    return v / v.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 성과
class PhaseData:
    """한 기간의 sel/out 및 행렬 캐시."""
    _store = {}

    def __new__(cls, pkey):
        if pkey not in cls._store:
            o = super().__new__(cls)
            o.pkey = pkey
            o.sel, o.out = build_phase(pkey)
            el = o.sel.index[o.sel["eligible"]]
            o.mon_eval = monthly_eval_matrix(pkey, el)
            hist = monthly_hist_matrix(pkey, el)
            o.hist_yield_m = hist.div(o.sel.loc[el, "base_price"], axis=0)
            cls._store[pkey] = o
        return cls._store[pkey]


def bundle_metrics(pd_: PhaseData, bundles: np.ndarray, w: np.ndarray, price_window=30,
                   prev_col="eval_prev_pay", pay_col="eval_pay") -> pd.DataFrame:
    sel, out = pd_.sel, pd_.out
    nb, N = bundles.shape
    flat = bundles.ravel()
    g = lambda s: s.reindex(flat).to_numpy(dtype=float).reshape(nb, N)
    p0 = g(sel["base_price"])
    shares = w / p0                                     # 투자금 1 기준 보유 주식 수
    ev = g(out[pay_col]); pv = g(out[prev_col])
    pay = (shares * ev).sum(1)
    prev = (shares * pv).sum(1)
    dec = np.clip(1 - pay / prev, 0, None)
    # 보조: 곡별 감소율의 투자금 가중 평균
    song_dec = np.clip(1 - ev / np.where(pv == 0, np.nan, pv), 0, None)
    song_dec = np.nansum(w * np.nan_to_num(song_dec), axis=1)
    # 월별 지급 변동계수
    M = pd_.mon_eval.reindex(flat).to_numpy(dtype=float).reshape(nb, N, -1)
    mp = (shares[:, :, None] * M).sum(1)
    cv = mp.std(axis=1, ddof=1) / mp.mean(axis=1)
    # 총수익률 (가격 커버리지 재가중)
    if price_window == 30:
        b0 = p0
    else:
        b0 = g(out[f"base_price_{price_window}"])
    p1 = g(out[f"end_price_{price_window}"])
    have = np.isfinite(p1) & np.isfinite(b0)
    cov = (w * have).sum(1)
    song_tr = np.where(have, (np.nan_to_num(p1) - np.nan_to_num(b0) + ev) / np.where(have, b0, 1), 0)
    tr = (w * song_tr).sum(1) / np.where(cov > 0, cov, np.nan)
    tr = np.where(cov >= COVERAGE_MIN, tr, np.nan)
    # 묶음 구성 특성: Dollar Age, 최대 1곡 지급 비중 (기대 지급 = 주식수 × LTM)
    ltm = g(sel["ltm"]); age = g(sel["age"])
    exp_pay = shares * ltm
    sh = exp_pay / exp_pay.sum(1, keepdims=True)
    return pd.DataFrame({"yield": pay / 1.0, "decline": dec, "declined": pay < prev,
                         "song_dec_w": song_dec, "cv": cv, "total_return": tr, "coverage": cov,
                         "dollar_age": (sh * age).sum(1), "max_pay_share": sh.max(1),
                         "w_max": w.max(1)})


def summarize(m: pd.DataFrame) -> dict:
    q = lambda s, p: float(np.nanpercentile(s, p))
    tr = m["total_return"].dropna()
    return {
        "묶음수": len(m),
        "평균지급수익률": m["yield"].mean(), "지급수익률_중앙": m["yield"].median(),
        "지급수익률_p5": q(m["yield"], 5), "지급수익률_p95": q(m["yield"], 95),
        "지급감소발생비율": m["declined"].mean(),
        "평균지급감소율": m["decline"].mean(), "감소율_중앙": m["decline"].median(),
        "감소율_p5": q(m["decline"], 5), "감소율_p95": q(m["decline"], 95),
        "곡별감소율가중평균(보조)": m["song_dec_w"].mean(),
        "월별지급CV_평균": m["cv"].mean(),
        "총수익률_평균": tr.mean() if len(tr) else np.nan,
        "총수익률_중앙": tr.median() if len(tr) else np.nan,
        "총수익률_p5": q(tr, 5) if len(tr) else np.nan,
        "총수익률_p95": q(tr, 95) if len(tr) else np.nan,
        "손실확률(총수익률<0)": (tr < 0).mean() if len(tr) else np.nan,
        "총수익률_집계묶음수": int(len(tr)),
        "가격커버리지_평균": m["coverage"].mean(),
        "DollarAge_평균": m["dollar_age"].mean(),
        "최대1곡지급비중_평균": m["max_pay_share"].mean(),
        "최대1곡지급비중_p95": q(m["max_pay_share"], 95),
    }


def run(pkey: str, spec: dict, n_bundles=N_BUNDLES, price_window=30, return_raw=False,
        prev_col="eval_prev_pay", pay_col="eval_pay"):
    """전략 하나를 한 기간에 적용. 요약 dict (와 원자료) 반환."""
    P = PhaseData(pkey)
    elig = apply_A(P.sel, spec["A"])
    pool = build_pool(P.sel, elig, spec["B"], spec["N"])
    res = {"기간": pkey, "전략": spec_name(spec), "복잡도": complexity(spec),
           "적격곡수": len(elig), "추출후보곡수": len(pool)}
    need = 3 * spec["N"] if spec["B"]["code"] in ("B1",) else spec["N"]
    if len(pool) < max(need, spec["N"]):
        res.update({"실행가능": False, "비고": f"공급 곡 수 {len(pool)} < 필요 {need}"})
        return (res, None) if return_raw else res
    tag = f"{spec_name(spec)}|{pkey}"
    bundles, fails = sample_bundles(P.sel, pool, spec, tag, n_bundles)
    w = weights(P.sel, bundles, spec["E"], P.hist_yield_m)
    m = bundle_metrics(P, bundles, w, price_window, prev_col=prev_col, pay_col=pay_col)
    res.update({"실행가능": True, "제약미충족묶음": fails, "비고": ""})
    res.update(summarize(m))
    return (res, (bundles, w, m)) if return_raw else res


# ---------------------------------------------------------------- 판단 기준
TOL = 0.005  # 0.5%p


def choose(rows: list[dict], baseline_yield: float | dict) -> tuple[pd.DataFrame, int]:
    """6절 판단 기준. rows: run() 결과 목록. baseline_yield: 후보별 비교 B0 수익률(스칼라 또는 전략명→값).
    반환: 판정 열이 붙은 표, 선택된 행 위치."""
    df = pd.DataFrame(rows).reset_index(drop=True)
    by = (df["전략"].map(baseline_yield) if isinstance(baseline_yield, dict)
          else pd.Series(baseline_yield, index=df.index))
    df["비교B0수익률"] = by
    ok = df["실행가능"].fillna(False).astype(bool)
    df["조건2_수익률≥B0"] = ok & (df["평균지급수익률"] >= by - 1e-12)
    cand = df[df["조건2_수익률≥B0"]]
    if cand.empty:
        cand = df[ok]
        note = "조건2를 만족하는 후보가 없어 실행가능 후보 전체에서 선택"
    else:
        note = ""
    best = cand["평균지급감소율"].min()
    near = cand[cand["평균지급감소율"] <= best + TOL - 1e-12]
    simplest = near[near["복잡도"] == near["복잡도"].min()]
    pick = simplest.sort_values(["감소율_p95", "평균지급감소율"], kind="mergesort").index[0]
    df["최저감소율과차이(%p)"] = (df["평균지급감소율"] - best) * 100
    df["0.5%p이내"] = df.index.isin(near.index)
    df["선택"] = df.index == pick
    # 순위(판단 기준 적용 순서):
    #  1) 선택  2) 조건2 충족 & 0.5%p 이내: 복잡도 → 감소율 p95
    #  3) 조건2 충족 & 0.5%p 밖: 평균감소율  4) 조건2 불충족: 평균감소율  5) 실행불가
    grp = np.select([df["선택"], df["조건2_수익률≥B0"] & df["0.5%p이내"], df["조건2_수익률≥B0"], ok],
                    [0, 1, 2, 3], 4)
    k1 = np.where(grp == 1, df["복잡도"], 0)
    k2 = np.where(grp == 1, df["감소율_p95"], df["평균지급감소율"])
    order = pd.DataFrame({"g": grp, "k1": k1, "k2": k2}, index=df.index) \
        .sort_values(["g", "k1", "k2"], kind="mergesort").index
    df["순위"] = pd.Series(range(1, len(df) + 1), index=order)
    df.attrs["note"] = note
    return df, pick


def spec_json(s: dict) -> dict:
    return json.loads(json.dumps(s, default=str))
