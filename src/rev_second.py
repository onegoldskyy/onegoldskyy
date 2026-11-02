"""[2차 탐색(사후)] 추가 요청 A(오래된 곡 안의 특성)와 B(아티스트 인기도).

연령 결과를 본 뒤 정한 질문이다. 판정·"유의" 표현에 쓰지 않는다. 결과는 results/revision/second/ 아래에 쓴다.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import PHASES, ISSUE_DATE
from data_prep import load_trades, load_cohort
from engine import _rng, PhaseData
from features import _base
from rev_common import REV, REPS
from rev_fast import Uni, bundle_perf, rand_sets, ANN
from rev_strats import tercile_keep
from rev_features import FEATS
import rev_run_helpers as H

warnings.filterwarnings("ignore")
TAG = "2차 탐색(사후)"
OUT = REV / "second"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
TB = OUT / "tables"
PKS = ["P1", "P2", "P2b"]
YEAR = {"P1": 2024, "P2": 2025, "P2b": 2026}
B_BOOT, R_BOOT = 200, 500
LOG = [f"# [{TAG}] 추가 요청 A·B 분석 로그\n",
       "- 이 분석은 0.5·0.6단계의 곡 연령 결과를 본 뒤 정한 질문이다. 모든 결과는 '2차 탐색(사후)'이며 판정이나 '유의' 표현에 쓰지 않는다.",
       f"- seed 20261101, 묶음 3,000개(점추정), 아티스트 군집 부트스트랩 B={B_BOOT}·복제당 {R_BOOT}개(구간은 기술용)."]


def tsave(df, name):
    d = df.copy(); d.insert(0, "표시", TAG)
    d.to_csv(TB / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


# ================================================================ 인기 지표 (구성일 이전 정보만)
def popularity(pk, W, trades):
    """W: 곡×월 1주당 지급, trades: 체결표. 둘 다 구성 전 자료만 넣으면 결과가 같아야 한다."""
    P = PhaseData(pk); sel = P.sel; Ph = PHASES[pk]
    S = _base()["S"]
    ids = sel.index
    art = sel["singer"]
    master = pd.read_csv(load_cohort.__globals__["COHORT_CSV"].parent / "source_song_master_796.csv", encoding="utf-8-sig")
    master["singer_clean"] = master["singer"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    n_reg = master.groupby("singer_clean")["song_id"].nunique()
    ltm = W.reindex(index=ids, columns=Ph["ltm"]).sum(1, min_count=12)
    trans = S["pct_transmission"].reindex(index=ids, columns=[c for c in Ph["ltm"] if c in S["pct_transmission"].columns]).sum(1)
    b0 = Ph["base_date"]
    t = trades[(trades["date"] <= b0) & (trades["date"] > b0 - pd.Timedelta(days=90))]
    val = (t["volume"] * t["close_price_krw"]).groupby(t["song_id"]).sum().reindex(ids).fillna(0)
    mdays = t["date"].nunique()
    tdays = t.groupby("song_id")["date"].nunique().reindex(ids).fillna(0) / max(mdays, 1)
    f = pd.DataFrame({"singer": art, "ltm": ltm, "trans": trans, "val90": val, "tday": tdays}, index=ids)
    g = f.groupby("singer")
    out = pd.DataFrame(index=ids)
    out["POP1"] = f["singer"].map(g["trans"].sum())          # 아티스트 전송 지급 합계(1주당, 곡 수로 나누지 않음)
    out["POP2"] = f["singer"].map(n_reg).fillna(1)            # 뮤직카우 등록 곡 수(796곡 원장)
    out["POP3"] = f["singer"].map(g["val90"].sum())           # 직전 90일 거래대금 합계
    out["POP4"] = f["singer"].map(g["tday"].mean())           # 직전 90일 평균 거래일 비율
    out["P2_ltmsum"] = f["singer"].map(g["ltm"].sum())        # 추가요청 A의 P2: 아티스트 전체 곡 LTM 1주당 지급 합계
    return out


def pop_leak_test(pk, trades_all):
    B = _base()
    full = popularity(pk, B["W"], trades_all)
    cut = pd.Period(PHASES[pk]["form"], "M")
    Wt = B["W"][[c for c in B["W"].columns if pd.Period(c, "M") < cut]]
    tt = trades_all[trades_all["date"] <= PHASES[pk]["base_date"]]
    trunc = popularity(pk, Wt, tt)
    return np.allclose(full.to_numpy(float), trunc.to_numpy(float), equal_nan=True), full


trades = load_trades()
POP, leak = {}, []
for pk in PKS:
    ok, POP[pk] = pop_leak_test(pk, trades)
    leak.append({"기간": pk, "구성일 이후 자료 제거 후 재계산 일치": ok}); assert ok
tsave(pd.DataFrame(leak), "R6_인기지표_누출테스트")
pc = []
for pk in PKS:
    Cm = POP[pk][["POP1", "POP2", "POP3", "POP4"]].corr(method="spearman"); Cm.insert(0, "기간", pk)
    pc.append(Cm.reset_index().rename(columns={"index": "지표"}))
tsave(pd.concat(pc).round(3), "R6_인기지표_상관")
LOG += ["- 인기 지표: POP1 아티스트 전송 지급 합계(선정용 12개월, 1주당), POP2 등록 곡 수(796곡 원장, 등록일 정보가 없어 2023-09-19 일괄 등록으로 봄), "
        "POP3 기초가격 기준일 이전 90일 거래대금 합계, POP4 같은 기간 곡별 거래일 비율(분모 = 시장 전체 체결일 수)의 아티스트 평균. 아티스트 = singer 원문.",
        "- 누출 테스트: 구성월 이후 지급 열과 기준일 이후 체결을 지운 자료로 다시 계산해도 세 기간 모두 같음(표 R6_인기지표_누출테스트).",
        "- POP-X(외부 차트 등): 구성 시점 기록이 있는 외부 자료가 저장소에 없어 쓰지 않았다.",
        "- 1주당 지급은 곡마다 발행 주식 수가 달라 곡 사이 절대 크기 비교에 한계가 있다(POP1·P2 공통 한계)."]


def build(pk):
    U = H.build06(pk)
    p = POP[pk].loc[U.df["song_id"].to_numpy()]
    for c in p.columns:
        U.df[c] = p[c].to_numpy()
    a = U.arr
    U.df["next_dec"] = np.clip(1 - a["ev"] / np.where(a["pv_ltm"] > 0, a["pv_ltm"], np.nan), 0, None)
    U.df["tr1"] = (a["p1"] - a["p0"] + a["ev"]) / a["p0"]
    U.df["log_mult"] = np.log(U.df["f8_multiple"].where(U.df["f8_multiple"] > 0))
    U.df["year"] = YEAR[pk]
    return U


UNIV = {pk: build(pk) for pk in PKS}


def metr(U, S):
    p = bundle_perf(U, S)
    tr = p["tr"]
    ok = np.isfinite(tr)
    return {"dec": p["dec_ltm"].mean(), "p95": np.percentile(p["dec_ltm"], 95), "y": p["yield"].mean() * ANN[U.pk],
            "tr": np.nanmean(tr), "tr5": np.percentile(tr[ok], 5) if ok.any() else np.nan,
            "loss": (tr[ok] < 0).mean() if ok.any() else np.nan}


def rand_from(U, mask, tag, R):
    pos = np.flatnonzero(mask)
    if len(pos) < 20:
        return None
    return pos[rand_sets(len(pos), _rng(tag), R)]


def compare(U, filt, tag, R=REPS, boot=True):
    """기준(후보군 무작위) vs 필터 구성. filt(U) → bool 마스크(후보군 U 안에서)."""
    base = metr(U, rand_from(U, np.ones(len(U), bool), tag + "|base", R))
    m = filt(U)
    S = rand_from(U, m, tag + "|f", R)
    if S is None:
        return None
    f = metr(U, S)
    d = {k: f[k] - base[k] for k in base}
    d["n_keep"] = int(m.sum()); d["n_univ"] = len(U)
    if boot:
        g = _rng("boot|" + tag); bs = []
        for b in range(B_BOOT):
            Ub = U.boot(g)
            Sb0 = rand_from(Ub, np.ones(len(Ub), bool), f"{tag}|b{b}|0", R_BOOT); Sb1 = rand_from(Ub, filt(Ub), f"{tag}|b{b}|1", R_BOOT)
            if Sb1 is None:
                continue
            bs.append(metr(Ub, Sb1)["dec"] - metr(Ub, Sb0)["dec"])
        d["boot_lo"], d["boot_hi"] = np.percentile(bs, [2.5, 97.5])
    return d


# ================================================================ A-1 오래된 곡 안의 특성
def univ_s5(U):
    return U.subset(U.df["m1_age"].to_numpy() >= 7)


def univ_m1(U):
    return U.subset(tercile_keep(U.df["m1_age"], "low"))


A_FEATS = {c: (FEATS[c][0], FEATS[c][1]) for c in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "M2"]}
A_FILTERS = {f"{c} 나쁜 3분위 제외": (lambda col, bad: lambda U: tercile_keep(U.df[col], bad))(*v) for c, v in A_FEATS.items()}


def top_tercile(col):
    def f(U):
        v = U.df[col].to_numpy(float); return v >= np.nanquantile(v, 2 / 3)
    return f


A_FILTERS["P1 등록곡수 상위 3분위만"] = top_tercile("POP2")
A_FILTERS["P1 등록곡수 상위 3분위 제외"] = lambda U: ~top_tercile("POP2")(U)
A_FILTERS["P2 아티스트 LTM합 상위 3분위만"] = top_tercile("P2_ltmsum")
A_FILTERS["P2 아티스트 LTM합 상위 3분위 제외"] = lambda U: ~top_tercile("P2_ltmsum")(U)
A_FILTERS["연령 15년 이상만"] = lambda U: U.df["m1_age"].to_numpy() >= 15
A_FILTERS["연령 15년 미만만"] = lambda U: U.df["m1_age"].to_numpy() < 15
rows = []
for ulab, ufn in [("S5 (7년 이상)", univ_s5), ("M1 (가장 어린 3분위 제외)", univ_m1)]:
    for pk in PKS:
        Uu = ufn(UNIV[pk])
        for fl, fn in A_FILTERS.items():
            d = compare(Uu, fn, f"A|{ulab}|{fl}|{pk}")
            if d is None:
                rows.append({"후보군": ulab, "특성": fl, "기간": pk, "비고": "남는 곡 20곡 미만"}); continue
            rows.append({"후보군": ulab, "특성": fl, "기간": pk, "후보군곡수": d["n_univ"], "남은곡수": d["n_keep"],
                         "Δ평균감소율(%p)": d["dec"] * 100, "Δ감소율p95(%p)": d["p95"] * 100, "Δ지급수익률_연환산(%p)": d["y"] * 100,
                         "Δ총수익률평균(%p)": d["tr"] * 100, "Δ총수익률p5(%p)": d["tr5"] * 100, "Δ손실확률(%p)": d["loss"] * 100,
                         "부트95%_하한(Δ감소율)": d["boot_lo"] * 100, "부트95%_상한(Δ감소율)": d["boot_hi"] * 100})
R5 = pd.DataFrame(rows)
sg = R5.groupby(["후보군", "특성"])["Δ평균감소율(%p)"].apply(lambda s: "개선(−) 세 기간 일치" if (s < 0).all() else ("악화(+) 세 기간 일치" if (s > 0).all() else "부호 불일치"))
R5["세기간부호(Δ감소율)"] = R5.set_index(["후보군", "특성"]).index.map(sg)
tsave(R5.round(3), "R5_오래된곡_내부특성")
LOG += ["- A-1: 후보군(S5 = 연령 7년 이상, M1 = 가장 어린 3분위 제외; 모두 0.6단계 기본 조건 표본 안) 무작위 20곡 대비, 후보군 안에서 다시 계산한 "
        "'나쁜 3분위 제외' 필터(인기 지표는 상위 3분위만/제외, 연령은 15년 이상만/미만만) 구성. 비교는 같은 후보군 무작위와의 평균 차이."]

# ================================================================ A-2 연령 교호 회귀 / A-3·B-1 인기 vs 가격·지급
L = pd.concat([UNIV[pk].df.assign(기간=pk) for pk in PKS], ignore_index=True)
L["artist"] = pd.factorize(L["singer"])[0]
zc = lambda s: (s - s.mean()) / s.std()
reg = []
for code, (col, bad) in A_FEATS.items():
    d = L[["next_dec", "m1_age", col, "year", "artist"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d["x"] = d.groupby("year")[col].transform(zc); d["age_c"] = d["m1_age"] - 7
    m = smf.ols("next_dec ~ age_c + x + age_c:x + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["artist"]})
    for v, lab in [("x", "특성(표준화, 연령 7년에서의 효과)"), ("age_c:x", "연령×특성(연령 1년당 특성 효과 변화)"), ("age_c", "연령(7년 기준)")]:
        reg.append({"특성": code, "측정식": FEATS[code][2], "변수": lab, "계수": m.params[v], "군집표준오차": m.bse[v], "p값(참고)": m.pvalues[v], "관측치": int(m.nobs)})
for pcol, plab in [("POP2", "P1 등록곡수"), ("P2_ltmsum", "P2 아티스트 LTM합")]:
    d = L[["next_dec", "m1_age", pcol, "year", "artist"]].dropna().copy()
    d["x"] = d.groupby("year")[pcol].transform(lambda s: (s >= s.quantile(2 / 3)).astype(float)); d["age_c"] = d["m1_age"] - 7
    m = smf.ols("next_dec ~ age_c + x + age_c:x + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["artist"]})
    for v, lab in [("x", "상위 3분위(1/0, 연령 7년에서)"), ("age_c:x", "연령×상위3분위"), ("age_c", "연령(7년 기준)")]:
        reg.append({"특성": plab, "측정식": "상위 3분위 더미", "변수": lab, "계수": m.params[v], "군집표준오차": m.bse[v], "p값(참고)": m.pvalues[v], "관측치": int(m.nobs)})
tsave(pd.DataFrame(reg).round(5), "R5b_연령교호회귀")

# 인기 5분위 번호(구성일별)
for c in ["POP1", "POP2", "POP3", "POP4", "P2_ltmsum"]:
    L[c + "_q"] = L.groupby("year")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)


def pop_reg(df, pops, label):
    out = []
    for dep, dlab in [("log_mult", "log 가격배수"), ("next_dec", "다음 해 지급감소율")]:
        x = " + ".join(p + "_q" for p in pops)
        f = f"{dep} ~ {x} + m1_age + f1_stab + f5_trans + f6_bro + C(year)"
        d = df[[dep, "m1_age", "f1_stab", "f5_trans", "f6_bro", "year", "artist"] + [p + "_q" for p in pops]].replace([np.inf, -np.inf], np.nan).dropna()
        m = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["artist"]})
        for p in pops:
            out.append({"표본": label, "모형": "함께" if len(pops) > 1 else "하나씩", "지표": p, "종속변수": dlab,
                        "인기1분위상승시_변화": m.params[p + "_q"], "군집표준오차": m.bse[p + "_q"], "p값(참고)": m.pvalues[p + "_q"],
                        "관측치": int(m.nobs)})
    return out


r6a = []
for lab, df in [("0.6단계 기본 조건 전체", L), ("7년 이상", L[L["m1_age"] >= 7])]:
    for p in ["POP1", "POP2", "POP3", "POP4"]:
        r6a += pop_reg(df, [p], lab)
    r6a += pop_reg(df, ["POP1", "POP2", "POP3", "POP4"], lab)
R6a = pd.DataFrame(r6a)
W6 = R6a.pivot_table(index=["표본", "모형", "지표"], columns="종속변수", values="인기1분위상승시_변화").reset_index()
W6 = W6.rename(columns={"log 가격배수": "가격배수 변화(log, 1분위당)", "다음 해 지급감소율": "지급감소율 변화(1분위당)"})
W6["가격배수 변화(%, 근사)"] = (np.exp(W6["가격배수 변화(log, 1분위당)"]) - 1) * 100
W6["지급감소율 변화(%p)"] = W6["지급감소율 변화(1분위당)"] * 100
tsave(R6a.round(5), "R6a_인기_가격vs지급")
tsave(W6.round(4), "R6a2_인기_가격vs지급_나란히")
# A-3 (R5c): 오래된 곡 표본의 P1·P2 지표
r5c = []
for lab, df in [("7년 이상", L[L["m1_age"] >= 7]), ("가장 어린 3분위 제외", pd.concat([univ_m1(UNIV[pk]).df for pk in PKS], ignore_index=True))]:
    df = df.copy(); df["artist"] = pd.factorize(df["singer"])[0]
    for c in ["POP2", "P2_ltmsum"]:
        df[c + "_q"] = df.groupby("year")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    r5c += pop_reg(df, ["POP2", "P2_ltmsum"], lab)
tsave(pd.DataFrame(r5c).round(5), "R5c_인기_가격vs지급")

# ================================================================ B-2 인기 × 연령 6칸
L["POP1_hi"] = L.groupby("year")["POP1"].transform(lambda s: np.where(s >= s.median(), "인기 상", "인기 하"))
L["연령3"] = pd.cut(L["m1_age"], [-np.inf, 7, 15, np.inf], right=False, labels=["7년 미만", "7~15년", "15년 이상"]).astype(str)
R6b = L.groupby(["기간", "POP1_hi", "연령3"]).agg(곡수=("song_id", "size"), 가격배수_평균=("f8_multiple", "mean"),
                                                 가격배수_중앙=("f8_multiple", "median"), 다음해지급감소율_평균=("next_dec", "mean"),
                                                 총수익률_평균=("tr1", "mean")).reset_index()
R6b["다음해지급감소율_평균"] *= 100; R6b["총수익률_평균"] *= 100
R6b = R6b.rename(columns={"POP1_hi": "인기(POP1 중앙값 기준)", "다음해지급감소율_평균": "다음해지급감소율_평균(%)", "총수익률_평균": "총수익률_평균(%)"})
tsave(R6b.round(3), "R6b_인기x연령")

# ================================================================ B-3 묶음 구성 / B-4 총수익 분해
def decomp(U, S):
    a = U.arr; p0 = a["p0"][S]; p1 = a["p1"][S]; ev = a["ev"][S]; have = np.isfinite(p1); w = np.full(S.shape, 1 / S.shape[1])
    cov = (w * have).sum(1)
    pay = (w * np.where(have, ev / p0, 0)).sum(1) / cov; prc = (w * np.where(have, (np.nan_to_num(p1) - p0) / p0, 0)).sum(1) / cov
    keep = cov >= 0.8
    return np.nanmean(np.where(keep, pay, np.nan)), np.nanmean(np.where(keep, prc, np.nan))


r6c = []
for ulab, ufn in [("적격 후보 전체(0.6 기본 조건)", lambda U: U), ("7년 이상", univ_s5)]:
    for pk in PKS:
        Uu = ufn(UNIV[pk])
        Sb = rand_from(Uu, np.ones(len(Uu), bool), f"B|{ulab}|{pk}|base", REPS)
        bp, bpr = decomp(Uu, Sb); bm = metr(Uu, Sb)
        for pop in ["POP1", "POP2", "POP3", "POP4"]:
            for comp, fn in [("상위 3분위만", top_tercile(pop)), ("상위 3분위 제외", (lambda tp: lambda U: ~tp(U))(top_tercile(pop)))]:
                d = compare(Uu, fn, f"B|{ulab}|{pop}|{comp}|{pk}")
                S = rand_from(Uu, fn(Uu), f"B|{ulab}|{pop}|{comp}|{pk}|f", REPS)
                pay, prc = decomp(Uu, S)
                r6c.append({"후보군": ulab, "지표": pop, "구성": comp, "기간": pk, "남은곡수": d["n_keep"],
                            "Δ평균감소율(%p)": d["dec"] * 100, "Δ지급수익률_연환산(%p)": d["y"] * 100, "Δ총수익률평균(%p)": d["tr"] * 100,
                            "Δ손실확률(%p)": d["loss"] * 100, "부트95%_하한(Δ감소율)": d["boot_lo"] * 100, "부트95%_상한(Δ감소율)": d["boot_hi"] * 100,
                            "지급수익(%)": pay * 100, "가격수익(%)": prc * 100, "기준_지급수익(%)": bp * 100, "기준_가격수익(%)": bpr * 100,
                            "Δ가격수익(%p)": (prc - bpr) * 100, "Δ지급수익(%p,커버곡)": (pay - bp) * 100})
R6c = pd.DataFrame(r6c)
sg = R6c.groupby(["후보군", "지표", "구성"])["Δ평균감소율(%p)"].apply(lambda s: "개선(−) 일치" if (s < 0).all() else ("악화(+) 일치" if (s > 0).all() else "불일치"))
R6c["세기간부호(Δ감소율)"] = R6c.set_index(["후보군", "지표", "구성"]).index.map(sg)
sgp = R6c.groupby(["후보군", "지표", "구성"])["Δ가격수익(%p)"].apply(lambda s: "+ 일치" if (s > 0).all() else ("− 일치" if (s < 0).all() else "불일치"))
R6c["세기간부호(Δ가격수익)"] = R6c.set_index(["후보군", "지표", "구성"]).index.map(sgp)
tsave(R6c.round(3), "R6c_인기_묶음")
(OUT / "analysis_log_2차탐색.md").write_text("\n".join(LOG), encoding="utf-8")
print("done")

# ================================================================ 기간별 부호 확인 (인기 회귀를 구성 연도별로)
yr = []
for lab, df in [("0.6단계 기본 조건 전체", L), ("7년 이상", L[L["m1_age"] >= 7])]:
    for y in sorted(df["year"].unique()):
        d0 = df[df["year"] == y]
        for p in ["POP1", "POP2", "POP3", "POP4"]:
            for dep, dlab in [("log_mult", "log 가격배수"), ("next_dec", "다음 해 지급감소율")]:
                d = d0[[dep, p + "_q", "m1_age", "f1_stab", "f5_trans", "f6_bro", "artist"]].replace([np.inf, -np.inf], np.nan).dropna()
                m = smf.ols(f"{dep} ~ {p}_q + m1_age + f1_stab + f5_trans + f6_bro", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["artist"]})
                yr.append({"표본": lab, "구성연도": y, "지표": p, "종속변수": dlab, "인기1분위상승시_변화": m.params[p + "_q"], "p값(참고)": m.pvalues[p + "_q"]})
YR = pd.DataFrame(yr)
sg = YR.groupby(["표본", "지표", "종속변수"])["인기1분위상승시_변화"].apply(lambda s: "+ 일치" if (s > 0).all() else ("− 일치" if (s < 0).all() else "불일치"))
YR["세기간부호"] = YR.set_index(["표본", "지표", "종속변수"]).index.map(sg)
tsave(YR.round(5), "R6a3_인기회귀_연도별부호")
print(YR.groupby(["표본", "지표", "종속변수"])["세기간부호"].first().to_string())
