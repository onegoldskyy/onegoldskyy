"""원자료 적재: 지급자료(월별), cohort, 체결가격 캐시, 기준일 체결가 조회."""
import numpy as np
import pandas as pd
from config import (ROYALTY_CSV, PRICE_CSV, COHORT_CSV, MATCH_CSV, CACHE, PCT_COLS)

PRICE_COLS = ["date", "provider_song_id", "close_price_krw", "volume", "price_type",
              "pre_current_security_issue", "is_partial_collection_day"]


def load_cohort() -> pd.DataFrame:
    c = pd.read_csv(COHORT_CSV, encoding="utf-8-sig")
    c["release_date"] = pd.to_datetime(c["release_date"])
    c["singer_clean"] = c["singer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return c


def load_royalty(cohort_only: bool = True) -> pd.DataFrame:
    r = pd.read_csv(ROYALTY_CSV, encoding="utf-8-sig")
    if cohort_only:
        r = r[r["song_id"].isin(load_cohort()["song_id"])].copy()
    r["release_date"] = pd.to_datetime(r["release_date"])
    return r


def royalty_wide(r: pd.DataFrame) -> pd.DataFrame:
    """곡 × 월(ym) 1주당 지급액 표. 해당 월 행이 없으면 NaN."""
    return r.pivot(index="song_id", columns="ym", values="amount_per_share").sort_index(axis=1)


def source_amount_wide(r: pd.DataFrame) -> dict:
    """수익원별 월 지급액(= 지급액 × 비중/100) 표 사전."""
    out = {}
    for c in PCT_COLS:
        t = r.assign(v=r["amount_per_share"] * r[c] / 100.0)
        out[c] = t.pivot(index="song_id", columns="ym", values="v").sort_index(axis=1)
    return out


def load_trades() -> pd.DataFrame:
    """사용 조건을 만족하는 체결 종가만 (캐시 사용)."""
    cache = CACHE / "trades_filtered.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    parts, n_total = [], 0
    for ch in pd.read_csv(PRICE_CSV, usecols=PRICE_COLS, chunksize=300_000, encoding="utf-8-sig"):
        n_total += len(ch)
        m = ((ch["price_type"] == "traded_daily_close") & (ch["volume"] > 0) &
             (ch["close_price_krw"] > 0) & (ch["pre_current_security_issue"] == 0) &
             (ch["is_partial_collection_day"] == 0))
        parts.append(ch.loc[m, ["date", "provider_song_id", "close_price_krw", "volume"]])
    t = pd.concat(parts, ignore_index=True)
    t["date"] = pd.to_datetime(t["date"])
    t = t.rename(columns={"provider_song_id": "song_id"}).sort_values(["song_id", "date"])
    t.attrs["n_total_rows"] = n_total
    t.to_pickle(cache)
    (CACHE / "trades_meta.txt").write_text(f"total_rows={n_total}\nfiltered_rows={len(t)}\n")
    return t


def last_trade(trades: pd.DataFrame, ref: pd.Timestamp, window: int) -> pd.DataFrame:
    """기준일 당일 포함 이전 마지막 체결 중 경과일 0~window(달력일)인 것. song_id 인덱스."""
    t = trades[(trades["date"] <= ref) & (trades["date"] >= ref - pd.Timedelta(days=window))]
    last = t.groupby("song_id").tail(1).set_index("song_id")
    return pd.DataFrame({"price": last["close_price_krw"].astype(float),
                         "trade_date": last["date"],
                         "age_days": (ref - last["date"]).dt.days})
