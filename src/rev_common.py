"""[수정 보완] 석사논문 수정 보완 분석 공통 모듈.

기존 엔진·자료를 재사용한다. 결과는 results/revision/ 아래에만 쓴다(기존 결과 파일은 수정하지 않음).
원고 규칙 재구현의 기본 가정(원고에 정의가 없는 부분)은 아래 상수로 고정한다. 결과를 보고 바꾸지 않는다.
"""
import numpy as np
import pandas as pd
from config import RESULTS
from engine import PhaseData, bundle_metrics, summarize, _rng

REV = RESULTS / "revision"
for _d in ("", "tables", "figures", "logs"):
    (REV / _d).mkdir(parents=True, exist_ok=True)
N, POOL, REPS = 20, 60, 3000
K_DEFAULT = 10          # 다양화 검색횟수(원고 미정의 → 사전 가정)
AGE_BINS = [-np.inf, 7, 15, np.inf]
AGE_LABELS = ["<7년", "7~15년", "15년+"]


def hhi(labels: np.ndarray) -> float:
    """범주 비중의 허핀달 지수(Σ 비중²). 작을수록 분산."""
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float((p ** 2).sum())


def elig(pk):
    P = PhaseData(pk)
    return P, P.sel[P.sel["eligible"]].copy()


def fin_score(s: pd.DataFrame) -> pd.Series:
    """재무정보 점수 = 과거 지급수익률 순위(높을수록 좋음)와 과거 지급감소 순위(작을수록 좋음)의 동등가중(평균 순위). 작을수록 좋음."""
    return (s["hist_yield"].rank(ascending=False) + s["hist_decline"].rank(ascending=True)) / 2


def top_pool(score: pd.Series, n=POOL, ascending=True) -> pd.Index:
    return score.sort_values(ascending=ascending, kind="mergesort").index[:n]


def rand_sets(pool, tag, reps=REPS, n=N):
    rng = _rng(tag); ids = np.asarray(pool)
    return np.array([rng.choice(ids, n, replace=False) for _ in range(reps)])


def paired_diversify(pool, tag, cat: pd.Series, K=K_DEFAULT, reps=REPS, n=N):
    """반복 r마다 같은 난수열로 K개 대안집합을 만든다.
    기준(재무정보) = 첫 집합 S_r1, 다양화 = K개 중 범주 HHI 최소 집합(동률이면 먼저 만든 집합)."""
    rng = _rng(tag); ids = np.asarray(pool)
    base, div = np.empty((reps, n), dtype=ids.dtype), np.empty((reps, n), dtype=ids.dtype)
    for r in range(reps):
        cand = [rng.choice(ids, n, replace=False) for _ in range(K)]
        h = [hhi(cat.loc[c].to_numpy()) for c in cand]
        base[r] = cand[0]; div[r] = cand[int(np.argmin(h))]
    return base, div


def metrics(P, bundles, prev_col="eval_prev_pay"):
    return bundle_metrics(P, bundles, np.full(bundles.shape, 1 / bundles.shape[1]), prev_col=prev_col)


def summ(m):
    return summarize(m)
