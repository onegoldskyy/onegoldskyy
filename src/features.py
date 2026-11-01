"""4절 곡 단위 변수 생성.

build_phase(pkey) 는 (sel, out) 두 표를 돌려준다.
  sel : 구성 시점 정보만 담은 선정용 변수 (선정 단계에서는 이 표만 사용한다)
  out : 평가기간 결과 (지급, 기말가격). 성과 계산에서만 사용한다.
"""
import numpy as np
import pandas as pd
from config import PHASES, PCT_COLS, SOURCE_LABEL, ISSUE_DATE
from data_prep import (load_cohort, load_royalty, royalty_wide, source_amount_wide,
                       load_trades, last_trade)

_CACHE = {}


def _base():
    if not _CACHE:
        r = load_royalty()
        _CACHE["r"] = r
        _CACHE["W"] = royalty_wide(r)
        _CACHE["S"] = source_amount_wide(r)
        _CACHE["c"] = load_cohort().set_index("song_id")
        _CACHE["t"] = load_trades()
    return _CACHE


def decade_label(year: int) -> str:
    if year < 1990:
        return "1990년대 이전"
    return f"{(year // 10) * 10}년대"


def age_bin(a):
    return pd.cut(a, [-np.inf, 7, 15, np.inf], right=False, labels=["<7년", "7~15년", "15년+"])


def build_phase(pkey: str, price_window: int = 30):
    B = _base()
    P = PHASES[pkey]
    W, S, c, t = B["W"], B["S"], B["c"], B["t"]
    ids = c.index
    ltm_m, prev_m = W.loc[ids, P["ltm"]], W.loc[ids, P["prev"]]
    lty_cols = [m for m in P["lty"] if m in W.columns]
    lty_m = W.loc[ids, lty_cols]
    base = last_trade(t, P["base_date"], price_window).reindex(ids)

    sel = pd.DataFrame(index=ids)
    sel["singer"] = c["singer_clean"]
    sel["title"] = c["title"]
    sel["release_date"] = c["release_date"]
    sel["age"] = (P["form"] - c["release_date"]).dt.days / 365.25
    sel["age_bin"] = age_bin(sel["age"])
    sel["decade"] = c["release_date"].dt.year.map(decade_label)
    sel["ltm_complete"] = ltm_m.notna().all(axis=1)
    sel["prev_complete"] = prev_m.notna().all(axis=1)
    sel["ltm"] = ltm_m.sum(axis=1, min_count=12)
    sel["prev12"] = prev_m.sum(axis=1, min_count=12)
    sel["lty_months"] = lty_m.notna().sum(axis=1)
    sel["lty"] = lty_m.mean(axis=1) * 12  # 최근 36개월 중 가용분의 연환산 평균
    sel["base_price"] = base["price"]
    sel["base_age_days"] = base["age_days"]
    sel["hist_yield"] = sel["ltm"] / sel["base_price"]
    ratio = sel["ltm"] / sel["prev12"].replace(0, np.nan)
    sel["hist_decline"] = (1 - ratio).clip(lower=0).fillna(0)
    sel["ltm_lty"] = sel["ltm"] / sel["lty"].replace(0, np.nan)
    sel["stab"] = (sel["ltm_lty"] - 1).abs()
    sel["multiple"] = sel["base_price"] / sel["ltm"].replace(0, np.nan)
    # 수익원: 선정용 12개월 지급 가중 평균 비중
    for col in PCT_COLS:
        amt = S[col].loc[ids, P["ltm"]].sum(axis=1)
        sel["sh_" + col.replace("pct_", "")] = amt / sel["ltm"].replace(0, np.nan)
    shcols = ["sh_" + x.replace("pct_", "") for x in PCT_COLS]
    sel["main_source"] = sel[shcols].fillna(-1).idxmax(axis=1).map(
        lambda s: SOURCE_LABEL["pct_" + s[3:]])
    sel["broadcast_share"] = sel["sh_broadcast"].fillna(0)
    # 월별 지급수익률 이력 (E4 공분산용, 선정용 LTY 기간)
    sel["eligible"] = (sel["ltm_complete"] & sel["prev_complete"] & sel["base_price"].notna())
    # 소급자료 사용 개월 수 (현행 증권 발행일 이전 월)
    used = sorted(set(P["ltm"]) | set(P["prev"]) | set(lty_cols))
    sel.attrs["retro_months"] = [m for m in used if pd.Period(m) < pd.Period(ISSUE_DATE, "M")]
    sel.attrs["lty_cols"] = lty_cols

    out = pd.DataFrame(index=ids)
    ev = W.loc[ids, P["eval"]]
    out["eval_months_complete"] = ev.notna().all(axis=1)
    out["eval_pay"] = ev.sum(axis=1, min_count=1)
    out["eval_prev_pay"] = W.loc[ids, P["eval_prev"]].sum(axis=1, min_count=1)
    out["eval_prev_alt_pay"] = W.loc[ids, P["eval_prev_alt"]].sum(axis=1, min_count=1)
    ev_bro = S["pct_broadcast"].loc[ids, P["eval"]].sum(axis=1)
    prev_bro = S["pct_broadcast"].loc[ids, P["eval_prev"]].sum(axis=1)
    out["eval_pay_xbro"] = out["eval_pay"] - ev_bro
    out["eval_prev_pay_xbro"] = out["eval_prev_pay"] - prev_bro
    for w in (7, 30, 90):
        e = last_trade(t, P["end_date"], w).reindex(ids)
        out[f"end_price_{w}"] = e["price"]
    for w in (7, 90):
        b = last_trade(t, P["base_date"], w).reindex(ids)
        out[f"base_price_{w}"] = b["price"]
    return sel, out


def monthly_eval_matrix(pkey: str, ids) -> pd.DataFrame:
    """평가기간 월별 1주당 지급액 (곡 × 월)."""
    B = _base()
    return B["W"].loc[ids, PHASES[pkey]["eval"]]


def monthly_hist_matrix(pkey: str, ids) -> pd.DataFrame:
    """선정용 과거 월별 지급액 (LTY 기간, 구성 시점 이전 정보)."""
    B = _base()
    P = PHASES[pkey]
    cols = [m for m in P["lty"] if m in B["W"].columns]
    return B["W"].loc[ids, cols]
