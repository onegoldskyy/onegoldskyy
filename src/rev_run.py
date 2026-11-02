"""[수정 보완] 주 실행: R0b·R0c 판정, R1·R1b, R2·R2b, R3b·R3c.

사전등록(results/revision/사전등록.md)에 적은 전략·기준만 쓴다.
"""
import itertools
import pickle
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from config import CACHE, PHASES
from rev_common import REV, REPS
from rev_fast import Uni, bundle_perf, summarize_perf, ANN
from rev_strats import strategies_05, strategies_06, run_one, bootstrap, diff_stat, tercile_keep, _rand, fin_pool
from rev_features import FEATS

warnings.filterwarnings("ignore")
TB = REV / "tables"
PKS = ["P1", "P2", "P2b"]
VAL = ["P2", "P2b"]
B_BOOT, R_BOOT = 200, 500
LOG = []
t0 = time.time()


def feats_df():
    f = pd.read_csv(TB / "features_ex_ante.csv", encoding="utf-8-sig")
    return {pk: g.set_index("song_id") for pk, g in f.groupby("기간")}


FE = feats_df()


def build06(pk):
    U = Uni.build(pk)
    f = FE[pk].loc[U.df["song_id"].to_numpy()]
    for c in [v[0] for v in FEATS.values()] + ["hist_months36"]:
        U.df[c] = f[c].to_numpy()
    U.df["decade"] = f["decade"].to_numpy()
    return U.subset(U.df["hist_months36"].to_numpy() >= 36)


def run_family(strats, builder, family):
    """점추정(3,000개)과 짝 기준 성과."""
    pt, raw = {}, {}
    for pk in PKS:
        U = builder(pk)
        for name, (fn, ref) in strats.items():
            o, p, base = run_one(U, fn, f"{family}|{name}|{pk}")
            pt[(name, pk)] = summarize_perf(p, pk)
            pt[(name, pk)]["표본곡수"] = len(U)
            raw[(name, pk)] = (o, p, base)
    return pt, raw


def judge(strats, pt, raw, builder, family, holm=False):
    rows = []
    boots = {}
    for name, (fn, ref) in strats.items():
        if ref is None:
            continue
        rec = {"전략": name, "비교기준": ref if ref != "PAIRED" else "같은 후보군의 첫 집합(짝)"}
        conds = {}
        for pk in PKS:
            s = pt[(name, pk)]
            if ref == "PAIRED":
                rb = summarize_perf(raw[(name, pk)][2], pk)
            else:
                rb = pt[(ref, pk)]
            dd = (s["평균감소율"] - rb["평균감소율"]) * 100
            dy = (s["지급수익률_연환산"] - rb["지급수익률_연환산"]) * 100
            rec.update({f"{pk}_평균감소율(%)": s["평균감소율"] * 100, f"{pk}_기준_평균감소율(%)": rb["평균감소율"] * 100,
                        f"{pk}_Δ감소율(%p)": dd, f"{pk}_지급수익률(%)": s["지급수익률"] * 100,
                        f"{pk}_Δ수익률_연환산(%p)": dy, f"{pk}_감소발생(%)": s["감소발생비중"] * 100,
                        f"{pk}_감소율p95(%)": s["감소율_p95"] * 100, f"{pk}_총수익률평균(%)": s["총수익률_평균"] * 100,
                        f"{pk}_총수익률p5(%)": s["총수익률_p5"] * 100})
            conds[pk] = (dd < 0, dy >= -0.3)
        c1 = conds["P1"][0]; c2 = conds["P2"][0] and conds["P2b"][0]
        c3 = all(v[1] for v in conds.values())
        rec.update({"조건1_2024감소↓": c1, "조건2_2025·2026감소↓": c2, "조건3_수익률하락≤0.3%p": c3})
        heavy = any(k in name for k in ("최소분산", "평균-분산"))
        do_boot = not heavy or (c1 and c2 and c3)
        ps = {}
        for pk in VAL:
            if not do_boot:
                rec[f"{pk}_부트95%구간"] = "미계산(조건1~3 미충족, 무거운 QP 전략)"; ps[pk] = np.nan; continue
            U = builder(pk)
            ref_fn = None if ref == "PAIRED" else strats[ref][0]
            bd = bootstrap(U, fn, ref_fn, f"{family}|{name}|{pk}", B_BOOT, R_BOOT)
            boots[(name, pk)] = bd
            lo, hi = np.percentile(bd["d_dec"] * 100, [2.5, 97.5])
            rec[f"{pk}_부트95%_하한"] = lo; rec[f"{pk}_부트95%_상한"] = hi
            ps[pk] = min(1.0, 2 * min((bd["d_dec"] >= 0).mean(), (bd["d_dec"] <= 0).mean()))
            rec[f"{pk}_부트p"] = ps[pk]
        c4 = any(rec.get(f"{pk}_부트95%_상한", np.inf) < 0 for pk in VAL)
        rec["조건4_검증구간0미포함"] = c4
        rec["검증p(min×2)"] = min(1.0, 2 * np.nanmin([ps[v] for v in VAL])) if not all(np.isnan(list(ps.values()))) else np.nan
        rec["세기간부호일치(감소↓)"] = all(conds[pk][0] for pk in PKS)
        rows.append(rec)
        LOG.append(f"  - {family} {name}: 조건 {int(c1)}{int(c2)}{int(c3)}{int(c4)} ({time.time() - t0:.0f}s)")
    J = pd.DataFrame(rows)
    if holm:
        J["Holm보정p"] = np.nan
        for bl in ["B-a", "B-b"]:
            m = J["전략"].str.endswith(bl)
            p = J.loc[m, "검증p(min×2)"].to_numpy(float)
            order = np.argsort(p); adj = np.empty_like(p); mx = 0
            for rank, i in enumerate(order):
                mx = max(mx, min(1.0, (len(p) - rank) * p[i])); adj[i] = mx
            J.loc[m, "Holm보정p"] = adj
        J["판정"] = np.where(J["조건1_2024감소↓"] & J["조건2_2025·2026감소↓"] & J["조건3_수익률하락≤0.3%p"] & (J["Holm보정p"] < 0.05),
                           "재무가치 있음", "일관되지 않음")
    else:
        J["판정"] = np.where(J["조건1_2024감소↓"] & J["조건2_2025·2026감소↓"] & J["조건3_수익률하락≤0.3%p"] & J["조건4_검증구간0미포함"],
                           "재무가치 있음", "일관되지 않음")
    return J, boots


# ================================================================ 0.5단계
S05 = strategies_05()
pt05, raw05 = run_family(S05, Uni.build, "S05")
LOG.append(f"- 0.5단계 점추정 완료 ({time.time() - t0:.0f}s)")
J05, boots05 = judge(S05, pt05, raw05, Uni.build, "S05")
J05.round(3).to_csv(TB / "R0b_전략전체_판정표.csv", index=False, encoding="utf-8-sig")
# S9 보조: S9a(동일금액) 대비
s9 = []
for nm in [n for n in S05 if n.startswith("S9")]:
    for pk in PKS:
        a, b = pt05[(nm, pk)], pt05[("S9a S4b 동일금액", pk)]
        s9.append({"전략": nm, "기간": pk, "Δ감소율_vs_S9a(%p)": (a["평균감소율"] - b["평균감소율"]) * 100,
                   "Δ수익률_vs_S9a(%p,연환산)": (a["지급수익률_연환산"] - b["지급수익률_연환산"]) * 100})
pd.DataFrame(s9).round(3).to_csv(TB / "R0b2_S9_동일금액대비.csv", index=False, encoding="utf-8-sig")

# ================================================================ 0.6단계
S06 = strategies_06()
pt06, raw06 = run_family(S06, build06, "S06")
LOG.append(f"- 0.6단계 점추정 완료 ({time.time() - t0:.0f}s)")
J06, boots06 = judge(S06, pt06, raw06, build06, "S06", holm=True)
passed = {bl: [n.split("@")[0] for n in J06.loc[(J06["판정"] == "재무가치 있음") & J06["전략"].str.endswith(bl), "전략"]] for bl in ["B-a", "B-b"]}
combo_rows = []
for bl, codes_ in passed.items():
    fil = [c for c in codes_ if c in FEATS]
    for a, b in itertools.combinations(fil, 2):
        ca, cb = FEATS[a], FEATS[b]
        def fn(U, g, R, ca=ca, cb=cb, bl=bl):
            keep = tercile_keep(U.df[ca[0]], ca[1]) & tercile_keep(U.df[cb[0]], cb[1])
            return {"S": _rand(np.flatnonzero(keep), g, R) if bl == "B-a" else _rand(fin_pool(U, keep), g, R)}
        for pk in PKS:
            U = build06(pk); _, p, _ = run_one(U, fn, f"combo|{a}+{b}|{bl}|{pk}")
            s = summarize_perf(p, pk); r = pt06[(("B-a 무작위" if bl == "B-a" else "B-b 재무정보"), pk)]
            combo_rows.append({"조합": f"{a}+{b}@{bl}", "기간": pk, "Δ감소율(%p)": (s["평균감소율"] - r["평균감소율"]) * 100,
                               "Δ수익률_연환산(%p)": (s["지급수익률_연환산"] - r["지급수익률_연환산"]) * 100})
J06.round(4).to_csv(TB / "R0c_구성요소_판정표.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(combo_rows if combo_rows else [{"조합": "단독 통과 필터 특성이 2개 미만이라 조합 없음"}]).round(3).to_csv(
    TB / "R0c2_통과특성_2개조합.csv", index=False, encoding="utf-8-sig")
# 특성 간 상관
cr = []
for pk in PKS:
    U = build06(pk)
    X = U.df[[v[0] for v in FEATS.values()]].replace([np.inf, -np.inf], np.nan)
    C = X.corr(method="spearman")
    C.index = C.columns = list(FEATS.keys())
    cr.append(C.assign(기간=pk).reset_index().rename(columns={"index": "특성"}))
pd.concat(cr).round(3).to_csv(TB / "R0d_특성상관.csv", index=False, encoding="utf-8-sig")

# ================================================================ R1·R1b (원고 규칙 + 등록 전략, 기간별)
r1 = []
for (nm, pk), s in pt05.items():
    r1.append({"전략": nm, "기간": pk, "지급수익률(%)": s["지급수익률"] * 100, "지급수익률_연환산(%)": s["지급수익률_연환산"] * 100,
               "감소발생비중(%)": s["감소발생비중"] * 100, "평균감소율(%)": s["평균감소율"] * 100, "감소율p95(%)": s["감소율_p95"] * 100,
               "총수익률평균(%)": s["총수익률_평균"] * 100, "총수익률p5(%)": s["총수익률_p5"] * 100, "표본곡수": s["표본곡수"]})
R1 = pd.DataFrame(r1)
MANU = {"S0 무작위": "무작위", "S3 재무정보 기반": "재무정보 기반", "S6a S3+연령 다양화": "+ 연령 다양화",
        "S6b S3+수익원 다양화": "+ 수익원 분산", "S6c S3+아티스트 다양화": "+ 아티스트 분산",
        "S2 과거 감소 작은 곡": "과거 감소 작은 곡", "S1 과거 수익률 중심": "과거 수익률 중심"}
R1["원고규칙"] = R1["전략"].map(MANU).fillna("")
R1.round(3).to_csv(TB / "R1_기간별_규칙성과.csv", index=False, encoding="utf-8-sig")
rk = R1.pivot(index="전략", columns="기간", values="평균감소율(%)").rank()
rb = []
for a, b in [("P1", "P2"), ("P1", "P2b"), ("P2", "P2b")]:
    for lab, sub in [("전체 등록 전략", rk.index), ("원고 7개 규칙", [k for k in MANU])]:
        tau = kendalltau(rk.loc[sub, a], rk.loc[sub, b])
        rb.append({"구분": lab, "기간쌍": f"{a}-{b}", "Kendall τ": tau.statistic, "전략수": len(sub)})
sign = []
for nm in [n for n in S05 if n.startswith(("S6", "S7"))]:
    ds = [(summarize_perf(raw05[(nm, pk)][1], pk)["평균감소율"] - summarize_perf(raw05[(nm, pk)][2], pk)["평균감소율"]) * 100 for pk in PKS]
    sign.append({"다양화 규칙": nm, "Δ감소율_2024(%p)": ds[0], "Δ감소율_2025(%p)": ds[1], "Δ감소율_2026(%p)": ds[2],
                 "세기간_부호일치": (np.sign(ds) == np.sign(ds[0])).all(), "부호": "개선(−)" if all(d < 0 for d in ds) else ("악화(+)" if all(d > 0 for d in ds) else "혼재")})
R1b = pd.concat([rk.reset_index().assign(구분="기간별 평균감소율 순위(1=가장 낮음)"), pd.DataFrame(rb), pd.DataFrame(sign)], ignore_index=True)
R1b.round(3).to_csv(TB / "R1b_순위안정성.csv", index=False, encoding="utf-8-sig")

# ================================================================ R2·R2b 짝비교
def pair_corr(U, S):
    M = U.arr["ev_m"][S] / U.arr["p0"][S][:, :, None]
    z = (M - M.mean(-1, keepdims=True)) / M.std(-1, keepdims=True)
    z = np.nan_to_num(z); n, m = S.shape[1], M.shape[-1]
    tot = (z.sum(1) ** 2).sum(-1) / m - (z ** 2).sum((1, 2)) / m
    return tot / (n * (n - 1))


r2, r2b, dist = [], [], []
PAIRS = [n for n in S05 if n.startswith(("S6", "S7"))]
for fam, names, raw, bts, builder in [("S05", PAIRS, raw05, boots05, Uni.build),
                                       ("S06", [n for n in S06 if n.startswith(("M3", "M4", "M5"))], raw06, boots06, build06)]:
    for nm in names:
        for pk in PKS:
            o, p, base = raw[(nm, pk)]
            d = (p["dec_ltm"] - base["dec_ltm"]) * 100
            dist.append(pd.DataFrame({"전략": nm, "기간": pk, "Δ": d}))
            bd = bts.get((nm, pk))
            lo, hi = (np.percentile(bd["d_dec"] * 100, [2.5, 97.5]) if bd is not None else (np.nan, np.nan))
            r2.append({"전략": nm, "기간": pk, "평균Δ감소율(%p)": d.mean(), "중앙Δ(%p)": np.median(d), "Δ<0비율(%)": (d < 0).mean() * 100,
                       "평균Δ수익률(%p)": (p["yield"] - base["yield"]).mean() * 100,
                       "Δ감소발생(%p)": (p["declined_ltm"].astype(float) - base["declined_ltm"]).mean() * 100,
                       "군집부트95%_하한": lo, "군집부트95%_상한": hi,
                       "비고": "" if bd is not None else "선택기간(2024): 부트스트랩은 검증기간만 계산"})
            U = builder(pk); df = U.df; Sd, Sb = o["S"], o["base"]
            row = {"전략": nm, "기간": pk}
            for col, lab in [("hist_yield", "과거수익률"), ("hist_decline", "과거감소"), ("age", "곡연령"), ("broadcast_share", "방송비중")]:
                v = df[col].to_numpy(float)
                row[f"Δ편입곡평균_{lab}"] = (np.nanmean(v[Sd], 1) - np.nanmean(v[Sb], 1)).mean() * (100 if lab != "곡연령" else 1)
            row["Δ곡쌍평균상관(평가기간 월지급)"] = (pair_corr(U, Sd) - pair_corr(U, Sb)).mean()
            row["Δ감소율(%p)"] = d.mean()
            row["Δ곡평균감소율 성분(%p)"] = (p["song_dec_avg"] - base["song_dec_avg"]).mean() * 100
            row["Δ결합(공동변동) 성분(%p)"] = row["Δ감소율(%p)"] - row["Δ곡평균감소율 성분(%p)"]
            # 곡 평균 성과가 비슷한 짝만(|Δ곡평균감소율|<0.5%p) 다시 비교
            mm = np.abs(p["song_dec_avg"] - base["song_dec_avg"]) * 100 < 0.5
            row["곡평균성과 일치 짝 수"] = int(mm.sum())
            row["일치 짝의 Δ감소율(%p)"] = d[mm].mean() if mm.any() else np.nan
            r2b.append(row)
pd.DataFrame(r2).round(3).to_csv(TB / "R2_짝비교.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(r2b).round(4).to_csv(TB / "R2b_차이원인.csv", index=False, encoding="utf-8-sig")
pd.concat(dist).to_pickle(CACHE / "rev_pair_dist.pkl")

# ================================================================ R3b·R3c
r3b, r3c = [], []
for (nm, pk), (o, p, base) in raw05.items():
    tr = p["tr"]; ok = np.isfinite(tr)
    dcl, loss = p["declined_ltm"][ok], tr[ok] < 0
    r3b.append({"전략": nm, "기간": pk, "집계묶음": int(ok.sum()),
                "감소O·손실O(%)": (dcl & loss).mean() * 100, "감소O·손실X(%)": (dcl & ~loss).mean() * 100,
                "감소X·손실O(%)": (~dcl & loss).mean() * 100, "감소X·손실X(%)": (~dcl & ~loss).mean() * 100,
                "상관(감소율,총수익률)": np.corrcoef(p["dec_ltm"][ok], tr[ok])[0, 1] if ok.sum() > 2 else np.nan,
                "평균감소율(%)": p["dec_ltm"].mean() * 100, "손실확률(%)": loss.mean() * 100 if ok.any() else np.nan,
                "총수익률p5(%)": np.percentile(tr[ok], 5) * 100 if ok.any() else np.nan})
    for base_k, lab in [("ltm", "① 선정용 직전 12개월(원고)"), ("cal", "② 직전 달력연도"), ("lty", "③ 직전 36개월 연평균"), ("yoy", "④ 월별 전년동월 대비 평균")]:
        r3c.append({"전략": nm, "기간": pk, "기준기간": lab, "감소발생(%)": p[f"declined_{base_k}"].mean() * 100,
                    "평균감소율(%)": p[f"dec_{base_k}"].mean() * 100})
R3b = pd.DataFrame(r3b)
for pk in PKS:
    m = R3b["기간"] == pk
    R3b.loc[m, "순위_평균감소율"] = R3b.loc[m, "평균감소율(%)"].rank()
    R3b.loc[m, "순위_손실확률"] = R3b.loc[m, "손실확률(%)"].rank()
    R3b.loc[m, "순위_총수익률p5(높을수록1)"] = (-R3b.loc[m, "총수익률p5(%)"]).rank()
sp = [{"기간": pk, "Spearman(감소율순위, 손실확률순위)": spearmanr(g["순위_평균감소율"], g["순위_손실확률"]).statistic,
       "Spearman(감소율순위, 총수익p5순위)": spearmanr(g["순위_평균감소율"], g["순위_총수익률p5(높을수록1)"]).statistic}
      for pk, g in R3b.groupby("기간")]
R3b.round(3).to_csv(TB / "R3b_감소vs손실.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(sp).round(3).to_csv(TB / "R3b2_위험순위상관.csv", index=False, encoding="utf-8-sig")
R3c = pd.DataFrame(r3c)
R3c["순위_평균감소율"] = R3c.groupby(["기간", "기준기간"])["평균감소율(%)"].rank()
R3c.round(3).to_csv(TB / "R3c_기준기간_민감도.csv", index=False, encoding="utf-8-sig")
with open(CACHE / "rev_state.pkl", "wb") as f:
    pickle.dump({"pt05": pt05, "pt06": pt06, "boots05": boots05, "boots06": boots06, "passed": passed}, f)
LOG.append(f"- 전체 완료 ({time.time() - t0:.0f}s)")
(REV / "logs" / "run_log.md").write_text("\n".join(LOG), encoding="utf-8")
print("\n".join(LOG[-5:]))
print(J05[["전략", "조건1_2024감소↓", "조건2_2025·2026감소↓", "조건3_수익률하락≤0.3%p", "조건4_검증구간0미포함", "판정"]].to_string(index=False))
print(J06[["전략", "조건1_2024감소↓", "조건2_2025·2026감소↓", "조건3_수익률하락≤0.3%p", "Holm보정p", "판정"]].to_string(index=False))
