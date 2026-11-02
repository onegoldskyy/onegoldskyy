"""[3차 탐색(사후)] 곡 연령별 유동성(L1·L2)과 가격 기준 대박·폭락(L3).

질문: 국내 시장이 오래된 곡을 상대적으로 싸게 매기는 이유가 유동성(거래가 드묾)인가.
결과는 results/revision/third/ 아래에 쓴다. 표준오차는 아티스트 군집.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import PHASES
from data_prep import load_trades, load_cohort, load_royalty, last_trade
from engine import PhaseData
from rev_common import REV

warnings.filterwarnings("ignore")
TAG = "3차 탐색(사후)"
OUT = REV / "third"; TB = OUT / "tables"; TB.mkdir(parents=True, exist_ok=True)
BINS = [-np.inf, 5, 7, 10, 15, np.inf]
LABS = ["5년 미만", "5~7년", "7~10년", "10~15년", "15년 이상"]
PKS = ["P1", "P2", "P2b"]; YEAR = {"P1": 2024, "P2": 2025, "P2b": 2026}
LOG = [f"# [{TAG}] 분석 로그\n"]


def tsave(df, name):
    d = df.copy(); d.insert(0, "표시", TAG); d.to_csv(TB / f"{name}.csv", index=False, encoding="utf-8-sig"); return df


def abin(a):
    return pd.cut(a, BINS, right=False, labels=LABS).astype(str)


coh = load_cohort().set_index("song_id")
trades = load_trades()
roy = load_royalty()
units = roy.pivot(index="song_id", columns="ym", values="cnt_units")

# ================================================================ L1 곡-월 거래 활동
t = trades[(trades["date"] >= "2023-10-01") & (trades["date"] <= "2026-08-31")].copy()
t["ym"] = t["date"].dt.to_period("M"); t["val"] = t["volume"] * t["close_price_krw"]
grid = pd.MultiIndex.from_product([coh.index, pd.period_range("2023-10", "2026-08", freq="M")], names=["song_id", "ym"])
M = t.groupby(["song_id", "ym"]).agg(days=("date", "nunique"), val=("val", "sum")).reindex(grid, fill_value=0).reset_index()
mend = {}
for p in pd.period_range("2023-10", "2026-08", freq="M"):
    mend[p] = last_trade(trades, p.to_timestamp(how="end").normalize(), 30)["price"]
M["price_me"] = [mend[p].get(s, np.nan) for s, p in zip(M["song_id"], M["ym"])]
M["units"] = [units.at[s, str(p)] if (s in units.index and str(p) in units.columns) else np.nan for s, p in zip(M["song_id"], M["ym"])]
M["turnover"] = M["val"] / (M["price_me"] * M["units"])
M["age"] = (M["ym"].dt.to_timestamp() - coh.loc[M["song_id"], "release_date"].to_numpy()).dt.days / 365.25
M["연령구간"] = abin(M["age"]); M["year"] = M["ym"].dt.year; M["no_trade"] = (M["days"] == 0).astype(float)
M["artist"] = pd.factorize(coh.loc[M["song_id"], "singer_clean"].to_numpy())[0]


def l1(g):
    return pd.Series({"곡월수": len(g), "거래없는곡월비율(%)": g["no_trade"].mean() * 100, "월체결일수_중앙": g["days"].median(),
                      "월체결일수_p90": g["days"].quantile(.9), "월거래대금_중앙(원)": g["val"].median(), "월거래대금_p90(원)": g["val"].quantile(.9),
                      "회전율_중앙(%)": g["turnover"].median() * 100, "회전율계산곡월": int(g["turnover"].notna().sum())})


L1 = pd.concat([M.groupby(["year", "연령구간"]).apply(l1).reset_index(),
                M.groupby("연령구간").apply(l1).reset_index().assign(year="2023-10~2026-08")], ignore_index=True)
L1["연령구간"] = pd.Categorical(L1["연령구간"], LABS, ordered=True)
L1 = L1.sort_values(["year", "연령구간"], key=lambda s: s.astype(str) if s.name == "year" else s)
tsave(L1.round(3), "L1_연령별_거래활동")
M["age2"] = M["age"] ** 2
r = []
for dep, lab in [("no_trade", "거래 없음(0/1, 선형확률)"), ("days", "월 체결일수")]:
    m = smf.ols(f"{dep} ~ age + age2 + C(year)", data=M).fit(cov_type="cluster", cov_kwds={"groups": M["artist"]})
    for v in ["age", "age2"]:
        r.append({"종속변수": lab, "변수": v, "계수": m.params[v], "군집표준오차": m.bse[v], "p값": m.pvalues[v], "관측치": int(m.nobs)})
    for a in [5, 10, 15, 20]:
        r.append({"종속변수": lab, "변수": f"예측값(연령 {a}년, 2025)", "계수": m.predict(pd.DataFrame({"age": [a], "age2": [a * a], "year": [2025]}))[0],
                  "군집표준오차": np.nan, "p값": np.nan, "관측치": int(m.nobs)})
tsave(pd.DataFrame(r).round(6), "L1b_거래활동_연령회귀")
LOG += ["- L1: cohort 767곡 × 2023-10~2026-08 곡-월. 체결 = 기존 사용 조건(traded_daily_close, 거래량>0, 현행 증권 이후, 부분 수집일 제외). "
        "거래대금은 체결 금액 열이 없어 일별 체결 수량 × 종가로 추정. 회전율 = 월 거래대금 ÷ (월말 이전 30일 내 마지막 체결가 × 그달 cnt_units). "
        "cnt_units는 지급자료의 월별 권리 수량이며 '발행 주식 수'로 보았다(주식 수 원자료가 따로 없음). 연령 = (그달 1일 − 발매일)/365.25."]

# ================================================================ L2 유동성과 가격
F = pd.read_csv(REV / "tables" / "features_ex_ante.csv", encoding="utf-8-sig")
rows = []
for pk in PKS:
    P = PhaseData(pk); s = P.sel; b0 = PHASES[pk]["base_date"]
    w = trades[(trades["date"] <= b0) & (trades["date"] > b0 - pd.Timedelta(days=90))]
    md = w["date"].nunique()
    td = w.groupby("song_id")["date"].nunique() / md
    v90 = (w["volume"] * w["close_price_krw"]).groupby(w["song_id"]).sum()
    d = s[s["eligible"] & (s["ltm"] > 0)][["singer", "age", "stab", "broadcast_share", "base_price", "lty", "multiple"]].copy()
    d["tday90"] = td.reindex(d.index).fillna(0); d["logval90"] = np.log1p(v90.reindex(d.index).fillna(0))
    d["year"] = YEAR[pk]; d["song_id"] = d.index
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
D = D[np.isfinite(D["multiple"]) & np.isfinite(D["stab"])]
D["log_mult_lty"] = np.log(D["base_price"] / D["lty"]); D["log_mult"] = np.log(D["multiple"])
D["artist"] = pd.factorize(D["singer"])[0]
reg = []
for dep, lab in [("log_mult_lty", "log(가격 ÷ 3년 평균 연저작권료) [표 3-7 M4]"), ("log_mult", "log(가격배수, LTM 기준) [표 3-7 M1]")]:
    for spec, x in [("유동성 넣기 전", "age + stab + broadcast_share"), ("유동성 넣은 후", "age + stab + broadcast_share + tday90 + logval90")]:
        m = smf.ols(f"{dep} ~ {x} + C(year)", data=D).fit(cov_type="cluster", cov_kwds={"groups": D["artist"]})
        for v in ["age", "tday90", "logval90"]:
            if v in m.params:
                reg.append({"종속변수": lab, "모형": spec, "변수": {"age": "연령(년)", "tday90": "직전 90일 거래일 비율", "logval90": "log(1+직전 90일 거래대금)"}[v],
                            "계수": m.params[v], "군집표준오차": m.bse[v], "p값": m.pvalues[v], "관측치": int(m.nobs), "R2": m.rsquared})
L2 = pd.DataFrame(reg)
tsave(L2.round(5), "L2_유동성과_가격")
cor = D[["age", "tday90", "logval90"]].corr(method="spearman").round(3)
tsave(cor.reset_index().rename(columns={"index": "변수"}), "L2b_연령_유동성_상관")
LOG += ["- L2: 표 3-7과 같은 표본(세 구성일 적격 곡 중 LTM>0)·같은 통제(연령, 안정성 |LTM/LTY−1|, 방송 비중, 연도 고정효과)에 유동성 변수 추가. "
        "직전 90일 = 기초가격 기준일 당일 포함 이전 90일. 거래일 비율 분모 = 그 기간 시장 전체 체결일 수. 거래대금은 log(1+원)."]

# ================================================================ L3 가격 기준 대박·폭락 vs 저작권료 기준
pr = []
for pk in PKS:
    P = PhaseData(pk); s = P.sel; o = P.out
    e = s["eligible"] & o["end_price_30"].notna()
    ch = (o.loc[e, "end_price_30"] / s.loc[e, "base_price"] - 1)
    pr.append(pd.DataFrame({"기간": pk, "chg": ch, "age": s.loc[e, "age"]}))
PR = pd.concat(pr); PR["연령구간"] = abin(PR["age"])


def l3(g, col="chg"):
    x = g[col]
    return pd.Series({"곡수": len(x), "+50%이상(%)": (x >= 0.5).mean() * 100, "2배이상(%)": (x >= 1.0).mean() * 100,
                      "−50%이하(%)": (x <= -0.5).mean() * 100, "중앙값(%)": x.median() * 100,
                      "하위5%(%)": x.quantile(.05) * 100, "상위5%(%)": x.quantile(.95) * 100})


L3p = pd.concat([PR.groupby(["기간", "연령구간"]).apply(l3).reset_index(),
                 PR.groupby("연령구간").apply(l3).reset_index().assign(기간="세 기간 합산")], ignore_index=True)
# 저작권료 기준(요청서 정의 재계산): ltm_check ÷ prev_check − 1, 연령 = m1_age − 13/12, 세 구성일(LTM 2023·2024·2025) 합산
R = F[(F["ltm_check"] > 0) & (F["prev_check"] > 0)].copy()
R["chg"] = R["ltm_check"] / R["prev_check"] - 1; R["age"] = R["m1_age"] - 13 / 12; R["연령구간"] = abin(R["age"])
L3r = R.groupby("연령구간").apply(l3).reset_index().assign(기준="저작권료(LTM ÷ 직전 12개월, 2023~2025 합산, 767곡 전체)")
L3pa = L3p[L3p["기간"] == "세 기간 합산"].drop(columns="기간").assign(기준="가격(기초가격 → 평가기간 끝 30일 내 체결가, 세 기간 합산, 적격 곡)")
CMP = pd.concat([L3pa, L3r], ignore_index=True)
CMP["연령구간"] = pd.Categorical(CMP["연령구간"], LABS, ordered=True); CMP = CMP.sort_values(["기준", "연령구간"])
L3p["연령구간"] = pd.Categorical(L3p["연령구간"], LABS, ordered=True)
tsave(L3p.sort_values(["기간", "연령구간"]).round(2), "L3_가격기준_대박폭락")
tsave(CMP.round(2), "L3b_가격vs저작권료_비교")
given = pd.DataFrame({"연령구간": LABS, "요청서_+50%이상(%)": [5.0, 6.3, 6.9, 5.3, 6.5], "요청서_−50%이하(%)": [9.8, 3.0, 1.7, 0.9, 2.6]})
chk = given.merge(L3r[["연령구간", "+50%이상(%)", "−50%이하(%)", "곡수"]], on="연령구간")
tsave(chk.round(2), "L3c_저작권료기준_요청서값_대조")
LOG += ["- L3 가격 기준: 각 구성일 적격 곡(기초가격 30일 내)의 평가기간 끝(2024·2025년 말, 2026-08-31) 이전 30일 내 마지막 체결가 대비 변화율. 기말가격이 없는 곡은 제외.",
        "- L3 저작권료 기준: 요청서 정의(features_ex_ante의 ltm_check ÷ prev_check, 연령 = m1_age − 13/12)로 다시 계산해 요청서 값과 대조(L3c)."]
(OUT / "analysis_log_3차탐색.md").write_text("\n".join(LOG), encoding="utf-8")
print(L1[L1["year"] == "2023-10~2026-08"].round(2).to_string(index=False))
print(pd.DataFrame(r).round(5).to_string(index=False))
print(L2.round(4).to_string(index=False)); print(cor.to_string())
print(CMP.round(2).to_string(index=False)); print(chk.round(2).to_string(index=False))
print(L3p[L3p["기간"] != "세 기간 합산"].pivot_table(index="연령구간", columns="기간", values=["중앙값(%)", "−50%이하(%)"]).round(1).to_string())

# ---- 보조(사후): 두 유동성 변수의 상관이 높아(0.83) 하나씩 넣은 모형도 보고
one = []
for dep, lab in [("log_mult_lty", "log(가격 ÷ 3년 평균 연저작권료)"), ("log_mult", "log(가격배수, LTM 기준)")]:
    for v, vl in [("tday90", "직전 90일 거래일 비율"), ("logval90", "log(1+직전 90일 거래대금)")]:
        m = smf.ols(f"{dep} ~ age + stab + broadcast_share + {v} + C(year)", data=D).fit(cov_type="cluster", cov_kwds={"groups": D["artist"]})
        one.append({"종속변수": lab, "넣은 유동성 변수": vl, "연령 계수": m.params["age"], "연령 p값": m.pvalues["age"],
                    "유동성 계수": m.params[v], "유동성 p값": m.pvalues[v], "관측치": int(m.nobs)})
tsave(pd.DataFrame(one).round(5), "L2c_유동성_하나씩(보조)")
print(pd.DataFrame(one).round(4).to_string(index=False))
