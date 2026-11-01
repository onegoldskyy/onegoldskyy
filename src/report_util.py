"""표 저장용 공통 함수: 비율 열을 %로 변환해 CSV로 저장."""
import pandas as pd
from config import TABLES

PCT_COLS = ["평균지급수익률", "지급수익률_중앙", "지급수익률_p5", "지급수익률_p95", "지급감소발생비율",
            "평균지급감소율", "감소율_중앙", "감소율_p5", "감소율_p95", "곡별감소율가중평균(보조)",
            "총수익률_평균", "총수익률_중앙", "총수익률_p5", "총수익률_p95", "손실확률(총수익률<0)",
            "가격커버리지_평균", "최대1곡지급비중_평균", "최대1곡지급비중_p95", "비교B0수익률", "월별지급CV_평균"]


def pct_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ren = {}
    for c in d.columns:
        if c in PCT_COLS:
            d[c] = (d[c].astype(float) * 100).round(2)
            ren[c] = c + ("" if c == "월별지급CV_평균" else "(%)")
    for c in d.columns:
        if d[c].dtype == float and c not in PCT_COLS:
            d[c] = d[c].round(3)
    return d.rename(columns=ren)


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = pct_table(df)
    d.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    return d


SHOW = ["순위", "선택", "전략", "적격곡수", "추출후보곡수", "평균지급수익률(%)", "비교B0수익률(%)", "조건2_수익률≥B0",
        "평균지급감소율(%)", "최저감소율과차이(%p)", "0.5%p이내", "복잡도", "감소율_p95(%)", "지급감소발생비율(%)"]


def md(d: pd.DataFrame, cols=SHOW) -> str:
    cols = [c for c in cols if c in d.columns]
    return d.sort_values("순위")[cols].to_markdown(index=False) if "순위" in d else d[cols].to_markdown(index=False)
