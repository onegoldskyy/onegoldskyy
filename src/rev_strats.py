"""[수정 보완] 사전등록 전략 정의(0.5·0.6단계)와 아티스트 군집 부트스트랩.

각 전략 함수는 (U, rng, R) → {"S": R×n 위치 배열, "w": 비중 또는 None, "base": (짝 기준 S 또는 None)} 를 돌려준다.
"""
import numpy as np
import pandas as pd
from engine import _rng
from rev_common import N, POOL, REPS, K_DEFAULT
from rev_fast import Uni, rand_sets, k_sets, hhi_sets, pick_div, bundle_perf, summarize_perf, weights
from rev_features import FEATS


def fin_score(df):
    return (df["hist_yield"].rank(ascending=False) + df["hist_decline"].rank(ascending=True)) / 2


def topk(series, n=POOL, ascending=True):
    return np.argsort(series.to_numpy() * (1 if ascending else -1), kind="mergesort")[:n]


def codes(series):
    return pd.factorize(series.astype(str))[0]


# ------------------------------------------------------------ 기본 블록
def _rand(pool_pos, rng, R, n=N):
    pool_pos = np.asarray(pool_pos)
    return pool_pos[rand_sets(len(pool_pos), rng, R, n)]


def _div(U, pool_pos, rng, R, cat=None, score_fn=None, maximize=False, K=K_DEFAULT):
    pool_pos = np.asarray(pool_pos)
    S3 = pool_pos[k_sets(len(pool_pos), rng, K, R)]
    if cat is not None:
        sc = hhi_sets(codes(U.df[cat]), S3)
    else:
        sc = score_fn(U, S3)
    return pick_div(S3, sc, maximize)


def dollar_age(U, S3):
    a = U.arr
    e = a["ltm"][S3] / a["p0"][S3]
    return (e * a["age"][S3]).sum(-1) / e.sum(-1)


def fin_pool(U, mask=None):
    df = U.df if mask is None else U.df[mask]
    pos = np.flatnonzero(np.ones(len(U.df), bool) if mask is None else np.asarray(mask))
    return pos[topk(fin_score(df.reset_index(drop=True)))]


# ------------------------------------------------------------ 0.5단계 전략
def strategies_05():
    st = {}
    st["S0 무작위"] = (lambda U, g, R: {"S": _rand(np.arange(len(U)), g, R)}, None)
    st["S1 과거 수익률 중심"] = (lambda U, g, R: {"S": _rand(topk(U.df["hist_yield"], ascending=False), g, R)}, "S0 무작위")

    def s2(U, g, R):
        t = g.permutation(len(U))
        order = np.lexsort((t, U.df["hist_decline"].to_numpy()))[:POOL]
        return {"S": _rand(order, g, R)}
    st["S2 과거 감소 작은 곡"] = (s2, "S0 무작위")
    st["S3 재무정보 기반"] = (lambda U, g, R: {"S": _div(U, fin_pool(U), g, R, cat="singer")[0]}, "S0 무작위")
    for k, th in [("a", 0.1), ("b", 0.2), ("c", 0.3)]:
        st[f"S4{k} 안정성 ≤{th}"] = ((lambda th: lambda U, g, R: {"S": _rand(np.flatnonzero(U.df["stab"] <= th + 1e-12), g, R)})(th), "S0 무작위")
    st["S5 연령 ≥7년"] = (lambda U, g, R: {"S": _rand(np.flatnonzero(U.df["age"] >= 7), g, R)}, "S0 무작위")
    for k, cat, lab in [("a", "age_bin3", "연령"), ("b", "main_source", "수익원"), ("c", "singer", "아티스트")]:
        def s6(U, g, R, cat=cat):
            b, d = _div(U, fin_pool(U), g, R, cat=cat)
            return {"S": d, "base": b}
        st[f"S6{k} S3+{lab} 다양화"] = (s6, "PAIRED")
        def s7(U, g, R, cat=cat):
            b, d = _div(U, np.flatnonzero(U.df["stab"] <= 0.2 + 1e-12), g, R, cat=cat)
            return {"S": d, "base": b}
        st[f"S7{k} S4b+{lab} 다양화"] = (s7, "PAIRED")
    for k, n in [("a", 10), ("b", 30)]:
        st[f"S8{k} S4b {n}곡"] = ((lambda n: lambda U, g, R: {"S": _rand(np.flatnonzero(U.df["stab"] <= 0.2 + 1e-12), g, R, n)})(n), "S0 무작위")
    for k, kind, lab in [("a", "equal", "동일금액"), ("b", "invvol", "지급 역변동성"), ("c", "minvar", "최소분산"), ("d", "meanvar", "평균-분산")]:
        def s9(U, g, R, kind=kind):
            S = _rand(np.flatnonzero(U.df["stab"] <= 0.2 + 1e-12), g, R)
            return {"S": S, "w": weights(U, S, kind)}
        st[f"S9{k} S4b {lab}"] = (s9, "S0 무작위")
    return st


# ------------------------------------------------------------ 0.6단계 특성
def tercile_keep(series, bad):
    v = series.to_numpy(float)
    if bad == "binary":
        return v == 0
    x = np.where(np.isfinite(v), v, np.inf if bad == "high" else -np.inf)
    q = np.quantile(x[np.isfinite(x)], 2 / 3 if bad == "high" else 1 / 3)
    return x <= q if bad == "high" else x >= q


def strategies_06():
    st = {}
    st["B-a 무작위"] = (lambda U, g, R: {"S": _rand(np.arange(len(U)), g, R)}, None)
    st["B-b 재무정보"] = (lambda U, g, R: {"S": _rand(fin_pool(U), g, R)}, None)
    for code, (col, bad, desc) in FEATS.items():
        def fa(U, g, R, col=col, bad=bad):
            return {"S": _rand(np.flatnonzero(tercile_keep(U.df[col], bad)), g, R)}
        def fb(U, g, R, col=col, bad=bad):
            return {"S": _rand(fin_pool(U, tercile_keep(U.df[col], bad)), g, R)}
        st[f"{code}@B-a"] = (fa, "B-a 무작위"); st[f"{code}@B-b"] = (fb, "B-b 재무정보")
    for code, kw in [("M3", dict(score_fn=dollar_age, maximize=True)), ("M4", dict(cat="decade")), ("M5", dict(cat="singer"))]:
        def da(U, g, R, kw=kw):
            b, d = _div(U, np.arange(len(U)), g, R, **kw); return {"S": d, "base": b}
        def db(U, g, R, kw=kw):
            b, d = _div(U, fin_pool(U), g, R, **kw); return {"S": d, "base": b}
        st[f"{code}@B-a"] = (da, "PAIRED"); st[f"{code}@B-b"] = (db, "PAIRED")
    return st


# ------------------------------------------------------------ 실행·부트스트랩
def run_one(U, fn, tag, R=REPS):
    out = fn(U, _rng(tag), R)
    perf = bundle_perf(U, out["S"], out.get("w"))
    base = bundle_perf(U, out["base"]) if out.get("base") is not None else None
    return out, perf, base


def diff_stat(perf, ref):
    return {"d_dec": perf["dec_ltm"].mean() - ref["dec_ltm"].mean(), "d_yield": perf["yield"].mean() - ref["yield"].mean(),
            "d_occ": perf["declined_ltm"].mean() - ref["declined_ltm"].mean()}


def bootstrap(U, fn, ref_fn, tag, B=200, R=500):
    """아티스트 군집 부트스트랩. ref_fn=None이면 fn이 돌려주는 짝 기준(base)과 비교."""
    g = _rng(f"boot|{tag}")
    res = []
    for b in range(B):
        Ub = U.boot(g)
        o, p, base = run_one(Ub, fn, f"{tag}|b{b}", R)
        if ref_fn is None:
            ref = base
        else:
            _, ref, _ = run_one(Ub, ref_fn, f"{tag}|ref|b{b}", R)
        res.append(diff_stat(p, ref))
    return pd.DataFrame(res)
