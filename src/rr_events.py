"""[보조·사후 탐색] 역주행 사건 정의와 탐지.

전송 지급액(월) = amount_per_share × pct_transmission / 100
(pct_digital_audio_transmission은 cohort 767곡 전체에서 0이어서 합산해도 같다.)

기본 정의: 발매 3년 이상인 곡에서, 전송 지급액이 직전 12개월 평균(기준선)의 3배 이상인 달이
2개월 연속이면 사건. 사건 시작월은 첫 달. 연속 판정에는 첫 달 직전 12개월 기준선을 공통으로 쓴다.
기준선이 0이거나 전체 곡-월 기준선 분포의 하위 10%(분모가 작은 착시) 이하이면 제외한다.
같은 곡에서 직전 사건 시작월로부터 12개월 안에 다시 시작한 사건은 하나로 묶는다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from config import RESULTS
from data_prep import load_royalty, load_cohort

EXTRA = RESULTS / "extra"
for _d in ("tables", "figures", "logs"):
    (EXTRA / _d).mkdir(parents=True, exist_ok=True)

_C = {}


def transmission_wide() -> pd.DataFrame:
    """곡 × 월 전송 지급액(1주당)."""
    if "T" not in _C:
        r = load_royalty()
        r = r.assign(tr=r["amount_per_share"] * r["pct_transmission"] / 100.0)
        _C["T"] = r.pivot(index="song_id", columns="ym", values="tr").sort_index(axis=1)
        _C["meta"] = load_cohort().set_index("song_id")
    return _C["T"]


def baseline_frame():
    """각 곡-월의 직전 12개월 평균(12개월 모두 있어야 함)."""
    T = transmission_wide()
    base = T.T.rolling(12, min_periods=12).mean().shift(1).T  # 월 m 기준 m-12..m-1 평균
    return T, base


def detect(mult=3.0, consec=2, min_age=3.0, low_pct=0.10, merge_months=12):
    T, base = baseline_frame()
    meta = _C["meta"]
    cols = list(T.columns)
    per = pd.PeriodIndex(cols, freq="M")
    # 기준선 하위 10% 기준값: 기준선이 정의되는 모든 곡-월(기준선>0)의 분포
    b = base.to_numpy()
    valid_b = b[np.isfinite(b) & (b > 0)]
    low_cut = np.quantile(valid_b, low_pct)
    rows = []
    Tv = T.to_numpy()
    rel = meta.loc[T.index, "release_date"]
    for i, sid in enumerate(T.index):
        last_start = None
        for j in range(len(cols) - consec + 1):
            bj = b[i, j]
            if not np.isfinite(bj) or bj <= 0 or bj <= low_cut:
                continue
            age = (per[j].to_timestamp() - rel.iloc[i]).days / 365.25
            if age < min_age:
                continue
            seg = Tv[i, j:j + consec]
            if np.any(~np.isfinite(seg)) or np.any(seg < mult * bj):
                continue
            if last_start is not None and (per[j] - last_start).n < merge_months:
                continue  # 같은 사건(묶음)으로 처리
            last_start = per[j]
            rows.append({"song_id": sid, "title": meta.loc[sid, "title"], "singer": meta.loc[sid, "singer_clean"],
                         "release_date": rel.iloc[i].date(), "release_year": rel.iloc[i].year,
                         "event_ym": cols[j], "event_year": per[j].year, "age_at_event": age,
                         "baseline_12m": bj, "first_month_pay": seg[0], "mult_first": seg[0] / bj,
                         "mult_peak_consec": (seg / bj).max(), "mult_min_consec": (seg / bj).min(),
                         "mult_peak_6m": np.nanmax(Tv[i, j:j + 6]) / bj})
    ev = pd.DataFrame(rows)
    ev.attrs.update(dict(mult=mult, consec=consec, min_age=min_age, low_cut=float(low_cut)))
    return ev


def eligible_song_months(min_age=3.0, low_pct=0.10, consec=2):
    """사건이 될 수 있었던 곡-월 수(분모)와 곡 목록: 기준선 유효·연령 조건 충족."""
    T, base = baseline_frame()
    meta = _C["meta"]
    per = pd.PeriodIndex(T.columns, freq="M")
    b = base.to_numpy()
    low_cut = np.quantile(b[np.isfinite(b) & (b > 0)], low_pct)
    rel = meta.loc[T.index, "release_date"].to_numpy()
    month_start = per.to_timestamp().to_numpy()
    age = (month_start[None, :] - rel[:, None]) / np.timedelta64(1, "D") / 365.25
    ok = np.isfinite(b) & (b > low_cut) & (age >= min_age)
    ok[:, len(per) - consec + 1:] = False
    return pd.DataFrame(ok, index=T.index, columns=T.columns)
