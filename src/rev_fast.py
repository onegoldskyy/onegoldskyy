"""[수정 보완] 벡터화된 묶음 시뮬레이션 엔진.

정의는 기존 engine.py와 같다(동일금액 = 곡당 투자금 1/N, 보유 주식 수 = 투자금/기초가격,
총수익률은 기말가격 확보 곡으로 재가중하고 커버리지 80% 미만 묶음은 제외).
부트스트랩(아티스트 복원추출)에서 표본 전체를 다시 만들 수 있도록 곡 단위 배열을 위치(index)로 다룬다.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import PHASES, PCT_COLS
from engine import PhaseData, _rng, _ledoit_wolf
from features import _base
from rev_common import N, POOL, REPS, K_DEFAULT, AGE_BINS, AGE_LABELS

COV_MIN = 0.8
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}


class Uni:
    """한 기간의 표본(곡 집합)과 곡 단위 배열."""

    def __init__(self, df, arr, pk):
        self.df, self.arr, self.pk = df.reset_index(drop=True), arr, pk

    @classmethod
    def build(cls, pk, mask=None):
        P = PhaseData(pk); s = P.sel; o = P.out
        W = _base()["W"]
        m = s["eligible"] if mask is None else (s["eligible"] & mask(s))
        ids = s.index[m]
        s = s.loc[ids].copy(); o = o.loc[ids]
        Ph = PHASES[pk]
        p0 = s["base_price"].to_numpy(float)
        ev_m = W.loc[ids, Ph["eval"]].to_numpy(float)
        prev_m = W.loc[ids, [str(pd.Period(x) - 12) for x in Ph["eval"]]].to_numpy(float)
        lty_cols = [c for c in Ph["lty"] if c in W.columns]
        hist = W.loc[ids, lty_cols].to_numpy(float)
        nev = len(Ph["eval"])
        arr = dict(p0=p0, p1=o["end_price_30"].to_numpy(float), ev=o["eval_pay"].to_numpy(float),
                   pv_ltm=o["eval_prev_pay"].to_numpy(float), pv_cal=o["eval_prev_alt_pay"].to_numpy(float),
                   pv_lty=(s["lty"] * nev / 12).to_numpy(float), ev_m=ev_m, prev_m=prev_m, hist=hist,
                   ltm=s["ltm"].to_numpy(float), age=s["age"].to_numpy(float),
                   ev_xbro=o["eval_pay_xbro"].to_numpy(float))
        s["age_bin3"] = pd.cut(s["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
        s["song_id"] = s.index
        return cls(s, arr, pk)

    def __len__(self):
        return len(self.df)

    def take(self, pos):
        pos = np.asarray(pos)
        return Uni(self.df.iloc[pos], {k: v[pos] for k, v in self.arr.items()}, self.pk)

    def subset(self, mask):
        return self.take(np.flatnonzero(np.asarray(mask)))

    def boot(self, rng):
        """아티스트 군집 복원추출."""
        art = self.df["singer"].to_numpy()
        ua = np.unique(art)
        groups = {a: np.flatnonzero(art == a) for a in ua}
        pick = rng.choice(ua, len(ua), replace=True)
        return self.take(np.concatenate([groups[a] for a in pick]))


# ------------------------------------------------------------------ 추출
def rand_sets(n_pool, rng, R=REPS, n=N):
    return np.argsort(rng.random((R, n_pool)), axis=1)[:, :n]


def k_sets(n_pool, rng, K=K_DEFAULT, R=REPS, n=N):
    return np.stack([rand_sets(n_pool, rng, R, n) for _ in range(K)], axis=1)  # R × K × n


def hhi_sets(codes, S):
    """S: (..., n) 위치 배열, codes: 곡별 범주 코드 → 집합별 HHI."""
    c = codes[S]
    C = codes.max() + 1
    cnt = np.zeros(c.shape[:-1] + (C,))
    np.put_along_axis(cnt, c, 0, axis=-1)
    for j in range(c.shape[-1]):
        idx = c[..., j:j + 1]
        np.put_along_axis(cnt, idx, np.take_along_axis(cnt, idx, -1) + 1, axis=-1)
    p = cnt / c.shape[-1]
    return (p ** 2).sum(-1)


def pick_div(S3, score, maximize=False):
    """S3: R×K×n, score: R×K → 반복별 (첫 집합, 선택 집합)."""
    j = np.argmax(score, 1) if maximize else np.argmin(score, 1)
    return S3[:, 0, :], S3[np.arange(S3.shape[0]), j, :]


# ------------------------------------------------------------------ 성과
def bundle_perf(U, S, w=None, base="ltm"):
    """S: R×n 위치(풀 안의 위치가 아니라 U 안의 위치). w: R×n 비중(없으면 동일금액)."""
    a = U.arr
    R, n = S.shape
    w = np.full((R, n), 1 / n) if w is None else w
    sh = w / a["p0"][S]
    pay = (sh * a["ev"][S]).sum(1)
    pv = {"ltm": a["pv_ltm"], "cal": a["pv_cal"], "lty": a["pv_lty"]}
    out = {"yield": pay}
    for k, v in pv.items():
        prev = (sh * v[S]).sum(1)
        out[f"dec_{k}"] = np.clip(1 - pay / prev, 0, None)
        out[f"declined_{k}"] = pay < prev
    em = (sh[:, :, None] * a["ev_m"][S]).sum(1); pm = (sh[:, :, None] * a["prev_m"][S]).sum(1)
    out["dec_yoy"] = np.nanmean(np.clip(1 - em / np.where(pm > 0, pm, np.nan), 0, None), 1)
    out["declined_yoy"] = em.sum(1) < pm.sum(1)
    p1 = a["p1"][S]; p0 = a["p0"][S]; have = np.isfinite(p1)
    cov = (w * have).sum(1)
    tr = (w * np.where(have, (np.nan_to_num(p1) - p0 + a["ev"][S]) / p0, 0)).sum(1) / np.where(cov > 0, cov, np.nan)
    out["tr"] = np.where(cov >= COV_MIN, tr, np.nan)
    out["cov"] = cov
    # 곡 단위 평균 감소율(투자금 가중) — 차이 분해용
    sd = np.clip(1 - a["ev"][S] / np.where(a["pv_ltm"][S] > 0, a["pv_ltm"][S], np.nan), 0, None)
    out["song_dec_avg"] = np.nansum(w * np.nan_to_num(sd), 1)
    exp = sh * a["ltm"][S]
    out["dollar_age"] = (exp * a["age"][S]).sum(1) / exp.sum(1)
    return out


def summarize_perf(p, pk, base="ltm"):
    tr = p["tr"][np.isfinite(p["tr"])]
    return {"지급수익률": p["yield"].mean(), "지급수익률_연환산": p["yield"].mean() * ANN[pk],
            "감소발생비중": p[f"declined_{base}"].mean(), "평균감소율": p[f"dec_{base}"].mean(),
            "감소율_p95": np.percentile(p[f"dec_{base}"], 95),
            "총수익률_평균": tr.mean() if len(tr) else np.nan, "총수익률_p5": np.percentile(tr, 5) if len(tr) else np.nan,
            "손실확률": (tr < 0).mean() if len(tr) else np.nan, "총수익률집계": len(tr)}


# ------------------------------------------------------------------ 비중(S9)
def _qp(cov, lb, ub, mu=None, target=None):
    n = cov.shape[0]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if mu is not None:
        cons.append({"type": "ineq", "fun": lambda w: w @ mu - target})
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), jac=lambda w: 2 * cov @ w, method="SLSQP",
                   bounds=[(lb, ub)] * n, constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x if res.success else np.full(n, 1 / n), lb, ub)
    return w / w.sum()


def roll12(hist_y):
    """월 지급수익률(곡×월) → 겹치는 12개월 누적 지급수익률(곡×관측)."""
    h = np.nan_to_num(hist_y)
    cs = np.cumsum(h, 1)
    return cs[:, 11:] - np.concatenate([np.zeros((h.shape[0], 1)), cs[:, :-12]], 1)


def weights(U, S, kind):
    R, n = S.shape
    if kind == "equal":
        return np.full((R, n), 1 / n)
    hy = U.arr["hist"] / U.arr["p0"][:, None]
    if kind == "invvol":
        sd = np.nanstd(hy, 1, ddof=1)
        v = 1 / np.where(sd > 0, sd, np.nan)
        v = np.where(np.isfinite(v), v, np.nanmax(v))
        w = v[S]
        return w / w.sum(1, keepdims=True)
    Y12 = roll12(hy)
    out = np.empty((R, n))
    for r in range(R):
        X = Y12[S[r]].T
        cov = _ledoit_wolf(X)
        if kind == "minvar":
            out[r] = _qp(cov, 0.01, 0.15)
        elif kind == "meanvar":
            mu = X.mean(0)
            out[r] = _qp(cov, 0.01, 0.15, mu=mu, target=mu.mean() - 1e-12)
    return out
