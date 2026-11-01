"""4단계: 8절 추가 분석.
 3-5 단일곡 vs 묶음 / 3-6 바스켓 체결 가능성 / 3-7 안정성 프리미엄 / 3-8 방송 제외 공통위험
 3-9 표본편향 / 3-10 원고 대조
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import RESULTS, LOGS, CACHE, PHASES, N_BUNDLES, SEED, TABLES
from data_prep import load_trades, last_trade
from engine import run, apply_A, PhaseData, bundle_metrics, summarize, spec_name, _rng
from report_util import save

warnings.filterwarnings("ignore")
fz = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
FINAL = fz["final"]["spec"]
A0 = {"code": "A0", "dec_th": None, "stab_th": None}
B0R = {"A": A0, "B": {"code": "B0", "decade": None}, "C": "C0", "N": 20, "E": "E1"}
PKS = ["P1", "P2", "P2b"]
log = ["# 4단계. 추가 분석 (8절)\n", f"- 최종 전략: {fz['final']['name']}\n"]


def single(pk, A, pay_col="eval_pay", prev_col="eval_prev_pay"):
    P = PhaseData(pk)
    ids = apply_A(P.sel, A)
    return bundle_metrics(P, np.asarray(ids).reshape(-1, 1), np.ones((len(ids), 1)),
                          pay_col=pay_col, prev_col=prev_col)


# ------------------------------------------------------------ 1. 단일곡 vs 묶음
rows = []
for pk in PKS:
    s_final = summarize(single(pk, FINAL["A"]))
    s_all = summarize(single(pk, A0))
    b = run(pk, FINAL)
    r0 = run(pk, B0R)
    for lab, s in [("단일곡(최종 적격기준)", s_final), ("단일곡(전체 적격)", s_all),
                   ("B0무작위 20곡", r0), ("최종 묶음", b)]:
        rows.append({"기간": pk, "구분": lab, "표본(곡/묶음)": s["묶음수"],
                     "평균지급수익률": s["평균지급수익률"], "지급수익률_p5": s["지급수익률_p5"],
                     "지급감소발생비율": s["지급감소발생비율"], "평균지급감소율": s["평균지급감소율"],
                     "감소율_p95": s["감소율_p95"], "손실확률(총수익률<0)": s["손실확률(총수익률<0)"],
                     "총수익률_p5": s["총수익률_p5"], "총수익률_집계수": s["총수익률_집계묶음수"]})
t1 = pd.DataFrame(rows)
# 개선 폭: 최종 묶음 − 같은 적격기준 단일곡
imp = []
for pk in PKS:
    g = t1[t1["기간"] == pk].set_index("구분")
    a, b = g.loc["단일곡(최종 적격기준)"], g.loc["최종 묶음"]
    imp.append({"기간": pk, "구분": "개선폭(묶음−단일곡, %p)",
                **{c: (b[c] - a[c]) for c in ["평균지급수익률", "지급수익률_p5", "지급감소발생비율", "평균지급감소율",
                                              "감소율_p95", "손실확률(총수익률<0)", "총수익률_p5"]}})
t1 = pd.concat([t1, pd.DataFrame(imp)], ignore_index=True)
d1 = save(t1, "3-5_단일vs묶음")
log += ["## 1. 단일곡 vs 묶음\n", "- 단일곡 분포는 적격 곡 각각을 1곡 보유한 경우다. 개선폭 행은 최종 묶음에서 같은 적격기준 단일곡을 뺀 값(%p)이다.",
        d1.to_markdown(index=False), ""]

# ------------------------------------------------------------ 2. 바스켓 체결 가능성
t = load_trades()
bk = []
for pk in PKS:
    P = PhaseData(pk)
    sel = P.sel
    d0 = PHASES[pk]["base_date"]
    has = {w: set(last_trade(t, d0, w).index) for w in (7, 30)}
    hist_ok = sel["ltm_complete"] & sel["prev_complete"]
    pool_np = sel.index[hist_ok & (sel["stab"] <= FINAL["A"]["stab_th"] + 1e-12)]  # 가격조건 없는 최종 기준 후보군
    elig = apply_A(sel, FINAL["A"])
    # (a) 최종 전략 실제 묶음 3,000개(적격=30일 내 체결)에서 20곡 모두 7일 내 체결
    _, (bundles, _, _) = run(pk, FINAL, return_raw=True)
    in7 = np.vectorize(lambda x: x in has[7])(bundles)
    # (b) 가격조건 없는 후보군에서 무작위 20곡
    rng = _rng(f"basket|{pk}")
    ids = np.asarray(pool_np)
    sims = np.array([rng.choice(ids, 20, replace=False) for _ in range(N_BUNDLES)])
    all7 = np.vectorize(lambda x: x in has[7])(sims).all(1).mean()
    all30 = np.vectorize(lambda x: x in has[30])(sims).all(1).mean()
    bk.append({"기간": pk, "구성기준일": d0.date(),
               "cohort중_7일내체결비율": len(has[7] & set(sel.index)) / len(sel),
               "cohort중_30일내체결비율": len(has[30] & set(sel.index)) / len(sel),
               "최종적격후보중_7일내체결비율": np.mean([i in has[7] for i in elig]),
               "최종묶음_20곡모두7일내체결확률": in7.all(1).mean(),
               "최종묶음_7일내체결곡수_평균": in7.sum(1).mean(),
               "가격무조건후보군_곡수": len(ids),
               "가격무조건후보군_7일내체결비율": np.mean([i in has[7] for i in ids]),
               "가격무조건후보군_30일내체결비율": np.mean([i in has[30] for i in ids]),
               "가격무조건_20곡모두7일내확률": all7, "가격무조건_20곡모두30일내확률": all30})
t2 = pd.DataFrame(bk)
pc = [c for c in t2.columns if "비율" in c or "확률" in c]
t2[pc] = (t2[pc] * 100).round(2)
t2 = t2.rename(columns={c: c + "(%)" for c in pc})
t2.to_csv(TABLES / "3-6_바스켓_체결가능성.csv", index=False, encoding="utf-8-sig")
# 월말별 시계열 (cohort 767곡 중 최근 7·30일 내 체결 비율)
ms = []
cohort_ids = set(PhaseData("P2").sel.index)
for d in pd.date_range("2023-10-31", "2026-08-31", freq="ME"):
    r7 = len(set(last_trade(t, d, 7).index) & cohort_ids) / len(cohort_ids)
    r30 = len(set(last_trade(t, d, 30).index) & cohort_ids) / len(cohort_ids)
    ms.append({"월말": d.date(), "7일내체결비율(%)": round(r7 * 100, 2), "30일내체결비율(%)": round(r30 * 100, 2),
               "독립가정_20곡모두7일내(%)": round(r7 ** 20 * 100, 4)})
t2m = pd.DataFrame(ms)
t2m.to_csv(TABLES / "3-6b_월말별_체결비율.csv", index=False, encoding="utf-8-sig")
log += ["## 2. 가격 번들(바스켓) 실행 가능성\n",
        "- 체결 = 사용 조건을 만족한 체결 종가가 기준일 당일 포함 이전 7(30)일 내 존재.",
        "- '최종묶음'은 적격(30일 내 체결) 후보에서 만든 실제 3,000개 묶음이다. '가격무조건 후보군'은 지급이력·안정성 기준만 만족하고 가격 조건을 두지 않은 곡들이며, "
        "여기서 20곡을 무작위로 뽑아 모두 체결이 있었을 확률을 계산했다.",
        t2.to_markdown(index=False),
        f"\n- 월말별(2023-10~2026-08) cohort 7일 내 체결 비율: 평균 {t2m['7일내체결비율(%)'].mean():.1f}%, "
        f"범위 {t2m['7일내체결비율(%)'].min():.1f}~{t2m['7일내체결비율(%)'].max():.1f}%. 30일: 평균 {t2m['30일내체결비율(%)'].mean():.1f}%.\n"]

# ------------------------------------------------------------ 3. 안정성 프리미엄
frames = []
for pk in PKS:
    s = PhaseData(pk).sel
    s = s[s["eligible"] & (s["ltm"] > 0)].copy()
    s["year"] = PHASES[pk]["form"].year
    s["song_id"] = s.index
    frames.append(s)
R = pd.concat(frames, ignore_index=True)
R = R[np.isfinite(R["multiple"]) & np.isfinite(R["stab"])].copy()
R["log_multiple"] = np.log(R["multiple"])
R["log_multiple_lty"] = np.log(R["base_price"] / R["lty"])
R["stab_down"] = (1 - R["ltm_lty"]).clip(lower=0)
R["stab_up"] = (R["ltm_lty"] - 1).clip(lower=0)
R["artist"] = R["singer"].astype("category").cat.codes
lo, hi = R["multiple"].quantile([0.01, 0.99])
R["multiple_w"] = R["multiple"].clip(lo, hi)
specs = {
    "M1 log배수 ~ |LTM/LTY−1|": "log_multiple ~ stab + age + broadcast_share + C(year)",
    "M2 log배수 ~ 감소측·증가측": "log_multiple ~ stab_down + stab_up + age + broadcast_share + C(year)",
    "M3 배수(1·99% 절단) ~ |LTM/LTY−1|": "multiple_w ~ stab + age + broadcast_share + C(year)",
    "M4 log(가격/LTY) ~ |LTM/LTY−1|": "log_multiple_lty ~ stab + age + broadcast_share + C(year)",
}
reg = []
for nm, f in specs.items():
    m = smf.ols(f, data=R).fit(cov_type="cluster", cov_kwds={"groups": R["artist"]})
    for v in m.params.index:
        if v == "Intercept" or v.startswith("C(year)"):
            continue
        reg.append({"모형": nm, "변수": v, "계수": m.params[v], "군집표준오차": m.bse[v], "t": m.tvalues[v],
                    "p값": m.pvalues[v], "관측치": int(m.nobs), "군집(아티스트)수": R["artist"].nunique(), "R2": m.rsquared})
t3 = pd.DataFrame(reg).round(4)
t3.to_csv(TABLES / "3-7_안정성프리미엄_회귀.csv", index=False, encoding="utf-8-sig")
# 연령 구간별 배수
R["RE연령구간"] = pd.cut(R["age"], [-np.inf, 3, 10, np.inf], right=False, labels=["0~3년", "3~10년", "10년+"])
R["본연구연령구간"] = R["age_bin"].astype(str)
ab = []
for col in ["RE연령구간", "본연구연령구간"]:
    for (yr, k), g in R.groupby(["year", col], observed=True):
        ab.append({"구간체계": col, "구성연도": yr, "연령구간": k, "곡수": len(g), "배수_중앙": g["multiple"].median(),
                   "배수_p25": g["multiple"].quantile(.25), "배수_p75": g["multiple"].quantile(.75),
                   "|LTM/LTY−1|_중앙": g["stab"].median()})
    for k, g in R.groupby(col, observed=True):
        ab.append({"구간체계": col, "구성연도": "전체", "연령구간": k, "곡수": len(g), "배수_중앙": g["multiple"].median(),
                   "배수_p25": g["multiple"].quantile(.25), "배수_p75": g["multiple"].quantile(.75),
                   "|LTM/LTY−1|_중앙": g["stab"].median()})
t3b = pd.DataFrame(ab).round(3)
t3b.to_csv(TABLES / "3-7b_연령구간별_가격배수.csv", index=False, encoding="utf-8-sig")
# 안정성 5분위별 배수
R["안정도5분위"] = R.groupby("year")["stab"].transform(lambda x: pd.qcut(x, 5, labels=["Q1(가장 안정)", "Q2", "Q3", "Q4", "Q5(가장 불안정)"]))
t3c = R.groupby("안정도5분위", observed=True).agg(곡수=("multiple", "size"), 배수_중앙=("multiple", "median"),
                                                 log배수_평균=("log_multiple", "mean"), stab_중앙=("stab", "median")).round(3).reset_index()
t3c.to_csv(TABLES / "3-7c_안정도5분위별_가격배수.csv", index=False, encoding="utf-8-sig")
R[["year", "song_id", "multiple", "stab", "age", "broadcast_share", "RE연령구간", "본연구연령구간"]].to_pickle(CACHE / "multiple_panel.pkl")
log += ["## 3. 안정성 프리미엄 (가격배수 = 기초가격 ÷ LTM)\n",
        f"- 표본: 세 구성일(2024·2025·2026-01-01)의 적격 곡 중 LTM>0, 곡-연도 {len(R):,}개, 아티스트 군집 {R['artist'].nunique()}개. 연도 고정효과, 아티스트(singer 원문) 군집 표준오차.",
        "- M2의 감소측(LTM<LTY) 양(+)·증가측 음(−) 계수는 배수의 분모가 LTM이라 생기는 기계적 효과와 섞여 있다(LTM이 일시적으로 줄면 가격이 그대로여도 배수가 커진다). "
        "M4는 분모를 3년 평균(LTY)으로 바꿔 이 효과를 줄인 모형이다.",
        "- 여기서 배수는 1주 가격 ÷ 1주 연간 지급(= 지급수익률의 역수)이다. Royalty Exchange 경매 배수(권리 기간·수수료 구조가 다름)와 수준을 직접 비교할 수 없고, 방향만 비교한다.",
        t3.to_markdown(index=False), "", "### 연령 구간별 배수", t3b[t3b["구성연도"] == "전체"].to_markdown(index=False),
        "", "### 안정도 5분위별 배수", t3c.to_markdown(index=False), ""]

# ------------------------------------------------------------ 4. 방송 제외 공통위험
cr = []
for pk in PKS:
    for lab, spec in [("최종 묶음", FINAL), ("B0무작위 20곡", B0R)]:
        a = run(pk, spec)
        x = run(pk, spec, pay_col="eval_pay_xbro", prev_col="eval_prev_pay_xbro")
        cr.append({"기간": pk, "구분": lab, "평균지급감소율_전체": a["평균지급감소율"], "평균지급감소율_방송제외": x["평균지급감소율"],
                   "지급감소발생비율_전체": a["지급감소발생비율"], "지급감소발생비율_방송제외": x["지급감소발생비율"]})
    for lab, A in [("단일곡(최종 적격기준)", FINAL["A"])]:
        a = summarize(single(pk, A)); x = summarize(single(pk, A, "eval_pay_xbro", "eval_prev_pay_xbro"))
        cr.append({"기간": pk, "구분": lab, "평균지급감소율_전체": a["평균지급감소율"], "평균지급감소율_방송제외": x["평균지급감소율"],
                   "지급감소발생비율_전체": a["지급감소발생비율"], "지급감소발생비율_방송제외": x["지급감소발생비율"]})
    # 방송 몫 변화: 최종 적격 후보군 전체 합계 기준
    P = PhaseData(pk); el = apply_A(P.sel, FINAL["A"]); o = P.out.loc[el]; pr = P.sel.loc[el, "base_price"]
    tot_ev, tot_pv = (o["eval_pay"] / pr).sum(), (o["eval_prev_pay"] / pr).sum()
    x_ev, x_pv = (o["eval_pay_xbro"] / pr).sum(), (o["eval_prev_pay_xbro"] / pr).sum()
    cr.append({"기간": pk, "구분": "적격 후보군 전체(동일금액 합산)",
               "평균지급감소율_전체": max(0, 1 - tot_ev / tot_pv), "평균지급감소율_방송제외": max(0, 1 - x_ev / x_pv),
               "전체지급변화율": tot_ev / tot_pv - 1, "방송제외지급변화율": x_ev / x_pv - 1,
               "변화기여_방송몫(%p)": ((tot_ev - x_ev) - (tot_pv - x_pv)) / tot_pv,
               "변화기여_방송외(%p)": (x_ev - x_pv) / tot_pv,
               "비교기간_방송비중": 1 - x_pv / tot_pv, "평가기간_방송비중": 1 - x_ev / tot_ev,
               "곡별지급비율_중앙": float(np.median(o["eval_pay"] / o["eval_prev_pay"]))})
t4 = pd.DataFrame(cr)
t4["방송기여(감소율차,%p)"] = (t4["평균지급감소율_전체"] - t4["평균지급감소율_방송제외"]) * 100
t4["방송기여비중"] = np.where((t4["평균지급감소율_전체"] > 0) & ~t4["구분"].str.startswith("적격 후보군"),
                         (t4["평균지급감소율_전체"] - t4["평균지급감소율_방송제외"]) / t4["평균지급감소율_전체"], np.nan)
pc = [c for c in t4.columns if c not in ("기간", "구분", "방송기여(감소율차,%p)", "곡별지급비율_중앙")]
t4[pc] = (t4[pc].astype(float) * 100).round(2)
t4 = t4.rename(columns={c: c + ("" if c.endswith("(%p)") else "(%)") for c in pc})
t4["곡별지급비율_중앙"] = t4["곡별지급비율_중앙"].round(3)
t4["방송기여(감소율차,%p)"] = t4["방송기여(감소율차,%p)"].round(2)
t4.to_csv(TABLES / "3-8_방송제외_공통위험.csv", index=False, encoding="utf-8-sig")
log += ["## 4. 제거 불가능한 공통위험: 방송 몫 제외\n",
        "- 평가기간·비교기간 지급에서 방송 몫(지급액×방송비중)을 빼고 같은 묶음 규칙으로 다시 계산했다. "
        "방송기여비중 = (전체 감소율 − 방송제외 감소율) ÷ 전체 감소율.",
        "- '적격 후보군 전체' 행은 후보 곡을 모두 같은 금액으로 담은 묶음(분산 한계)의 지급 변화율을 방송 몫 기여와 방송 외 기여(%p, 합=전체 변화율)로 나눈 것이다.",
        "- 방송기여가 음수인 해는 방송 몫이 오히려 늘어 감소를 완화한 해다. 방송 정산은 해마다 방향이 바뀌는 공통 요인이다.",
        "- 주의: 곡별 지급 변화는 오른쪽 꼬리가 길다. 예컨대 2025년 최종 적격 후보 456곡의 곡별 지급비율(평가÷비교) 중앙값은 0.863인데, "
        "상위 5곡(최대 32.7배)이 합계를 9.8%p 끌어올려 후보군 합산 변화율은 +0.5%가 된다. 그래서 '후보군 전체 합산'은 감소가 없어 보여도 "
        "20곡 묶음의 다수는 감소한다.",
        t4.to_markdown(index=False), ""]

# ------------------------------------------------------------ 5. 표본편향 (30일 가격조건)
sb = []
for pk in PKS:
    P = PhaseData(pk); s = P.sel; o = P.out
    hist = s["ltm_complete"] & s["prev_complete"]
    kept = hist & s["base_price"].notna(); drop = hist & s["base_price"].isna()
    b90 = o["base_price_90"]
    ev_dec = (1 - o["eval_pay"] / o["eval_prev_pay"]).clip(lower=0)
    feats = {"곡연령(년)": s["age"], "LTM(1주,원)": s["ltm"], "과거지급감소율": s["hist_decline"],
             "|LTM/LTY−1|": s["stab"], "방송비중": s["broadcast_share"], "전송비중": s["sh_transmission"],
             "과거지급수익률(90일가격)": s["ltm"] / b90, "평가기간지급감소율(사후)": ev_dec,
             "2000년대비율": (s["decade"] == "2000년대").astype(float), "2010년대비율": (s["decade"] == "2010년대").astype(float),
             "2020년대비율": (s["decade"] == "2020년대").astype(float)}
    for k, v in feats.items():
        a, b = v[kept].dropna(), v[drop].dropna()
        sdp = np.sqrt((a.var() + b.var()) / 2)
        sb.append({"기간": pk, "변수": k, "남은곡수": len(a), "빠진곡수": len(b), "남은곡_평균": a.mean(), "빠진곡_평균": b.mean(),
                   "남은곡_중앙": a.median(), "빠진곡_중앙": b.median(),
                   "표준화차이": (b.mean() - a.mean()) / sdp if sdp > 0 else np.nan})
t5 = pd.DataFrame(sb).round(4)
t5.to_csv(TABLES / "3-9_표본편향_가격조건.csv", index=False, encoding="utf-8-sig")
log += ["## 5. 표본편향: 30일 가격조건으로 빠진 곡 vs 남은 곡\n",
        "- 빠진 곡 = 지급이력은 완전하지만 기초가격 30일 내 체결이 없는 곡. 과거지급수익률은 빠진 곡에 30일 가격이 없어 90일 내 체결가로 계산했다(가능한 곡만). "
        "평가기간 지급감소율은 사후 결과로, 편향 방향 확인용이다. 표준화차이 |d|≥0.2면 무시하기 어려운 차이로 본다.",
        t5.to_markdown(index=False), ""]

# ------------------------------------------------------------ 6. 원고 대조
t5ms = pd.read_csv(TABLES / "3-2_원고표5_재현.csv", encoding="utf-8-sig")
r0 = run("P2", B0R)
cmp = pd.DataFrame([
    {"항목": "표5 연령별 단일곡(2025) 전 항목", "원고": "164/315/183곡, 4.87/6.38/6.89%, 82.32/78.41/78.14%, 21.42/16.15/16.31%p",
     "본분석": f"{'/'.join(map(str, t5ms['재현_곡수']))}곡, {'/'.join(f'{x:.2f}' for x in t5ms['재현_평균지급수익률(%)'])}%, "
               f"{'/'.join(f'{x:.2f}' for x in t5ms['재현_지급감소곡비중(%)'])}%, {'/'.join(f'{x:.2f}' for x in t5ms['재현_평균감소율(%p)'])}%p",
     "일치": bool(t5ms.filter(like="일치_").all().all())},
    {"항목": "표7 전체 후보 무작위(2025) 평균 지급수익률", "원고": "6.22%", "본분석": f"{r0['평균지급수익률']:.2%} (B0무작위 20곡)", "일치": False},
    {"항목": "표7 전체 후보 무작위 지급감소 발생 비중", "원고": "77.47%", "본분석": f"{r0['지급감소발생비율']:.2%}", "일치": False},
    {"항목": "표7 전체 후보 무작위 전체 평균 감소율", "원고": "9.49%p", "본분석": f"{r0['평균지급감소율']:.2%}", "일치": False},
])
cmp.to_csv(TABLES / "3-10_원고대조.csv", index=False, encoding="utf-8-sig")
log += ["## 6. 기존 원고 대조\n", cmp.to_markdown(index=False),
        "\n- 표 5는 모든 칸이 일치한다. 표 7 무작위 기준선은 근사하지만 일치하지 않는다. 원고의 묶음 크기·추출 횟수·seed가 문서에 없어 같은 조건을 재구성할 수 없기 때문이며, "
        "원고 표 7은 이 분석의 판단 근거로 쓰지 않는다.\n"]

(LOGS / "04_추가분석.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
