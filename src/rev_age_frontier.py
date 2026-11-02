"""[수정 보완] R0e 곡 연령 곡선(판정 제외, 설명용)과 R4 효율적 프론티어 추가 기간."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from figures import BLUE, ORANGE, AQUA, INK, PH_COLOR  # 한글 폰트·스타일
from engine import _rng, _ledoit_wolf
from rev_common import REV, REPS
from rev_fast import Uni, bundle_perf, summarize_perf, rand_sets, roll12, _qp, ANN

warnings.filterwarnings("ignore")
TB, FG = REV / "tables", REV / "figures"
PKS = ["P1", "P2", "P2b"]
LAB = {"P1": "2024", "P2": "2025", "P2b": "2026.1~8"}
YEAR = {"P1": 2024, "P2": 2025, "P2b": 2026}

# ================================================================ R0e 연령 곡선
rows, songs = [], []
for pk in PKS:
    U = Uni.build(pk)
    q = pd.qcut(U.df["age"], 5, labels=False)
    for k in range(5):
        pos = np.flatnonzero(q == k)
        S = pos[rand_sets(len(pos), _rng(f"R0e|{pk}|{k}"), REPS)]
        s = summarize_perf(bundle_perf(U, S), pk)
        a = U.df["age"].iloc[pos]
        rows.append({"기간": pk, "연령5분위": k + 1, "연령범위(년)": f"{a.min():.1f}~{a.max():.1f}", "곡수": len(pos),
                     "평균감소율(%)": s["평균감소율"] * 100, "감소율p95(%)": s["감소율_p95"] * 100,
                     "지급수익률_연환산(%)": s["지급수익률_연환산"] * 100, "총수익률평균(%)": s["총수익률_평균"] * 100})
    d = U.df.copy()
    a = U.arr
    d["next_dec"] = np.clip(1 - a["ev"] / np.where(a["pv_ltm"] > 0, a["pv_ltm"], np.nan), 0, None)
    d["year"] = YEAR[pk]
    songs.append(d[["song_id", "singer", "age", "next_dec", "year"]])
R0e = pd.DataFrame(rows).round(3)
L = pd.concat(songs, ignore_index=True).dropna()
L["artist"] = pd.factorize(L["singer"])[0]
L["age2"] = L["age"] ** 2
L["RE구간"] = pd.cut(L["age"], [-np.inf, 3, 10, np.inf], right=False, labels=["0~3년", "3~10년", "10년+"]).astype(str)
L["원고구간"] = pd.cut(L["age"], [-np.inf, 7, 15, np.inf], right=False, labels=["<7년", "7~15년", "15년+"]).astype(str)
reg = []
for nm, f in [("연속: 연령+연령²", "next_dec ~ age + age2 + C(year)"),
              ("Royalty Exchange 구간(기준 3~10년)", "next_dec ~ C(RE구간, Treatment('3~10년')) + C(year)"),
              ("원고 구간(기준 7~15년)", "next_dec ~ C(원고구간, Treatment('7~15년')) + C(year)")]:
    m = smf.ols(f, data=L).fit(cov_type="cluster", cov_kwds={"groups": L["artist"]})
    for v in m.params.index:
        if v.startswith("C(year)") or v == "Intercept":
            continue
        reg.append({"모형": nm, "변수": v, "계수": m.params[v], "군집표준오차": m.bse[v], "p값": m.pvalues[v], "관측치": int(m.nobs)})
REG = pd.DataFrame(reg)
mc = REG[REG["모형"].str.startswith("연속")].set_index("변수")["계수"]
turn = -mc["age"] / (2 * mc["age2"]) if mc["age2"] != 0 else np.nan
bins = L.groupby(["year", "RE구간"])["next_dec"].agg(["size", "mean", "median"]).reset_index()
bins2 = L.groupby(["year", "원고구간"])["next_dec"].agg(["size", "mean", "median"]).reset_index()
R0e.to_csv(TB / "R0e_연령곡선.csv", index=False, encoding="utf-8-sig")
REG.round(5).to_csv(TB / "R0e2_연령회귀.csv", index=False, encoding="utf-8-sig")
pd.concat([bins.rename(columns={"RE구간": "구간"}).assign(구간체계="Royalty Exchange"),
           bins2.rename(columns={"원고구간": "구간"}).assign(구간체계="원고")]).round(4).to_csv(
    TB / "R0e3_연령구간별_곡단위감소율.csv", index=False, encoding="utf-8-sig")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, col in zip(axes, ["평균감소율(%)", "감소율p95(%)", "지급수익률_연환산(%)", "총수익률평균(%)"]):
    for pk in PKS:
        g = R0e[R0e["기간"] == pk]
        ax.plot(g["연령5분위"], g[col], color=PH_COLOR[pk], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=LAB[pk])
    ax.set_xticks(range(1, 6)); ax.set_xlabel("곡 연령 5분위(1=가장 어림)"); ax.set_title(col, loc="left")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("그림 R0e. 곡 연령 5분위별 20곡 묶음 성과 [수정보완·설명용, 판정 제외]", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.text(0.01, -0.05, f"주: 분위별 무작위 20곡 3,000개. 연속 회귀의 꺾이는 점(−b₁/2b₂) = {turn:.1f}년은 후속 검증 가설로만 본다.",
         fontsize=7.5, color="#5f5e5a")
fig.tight_layout(); fig.savefig(FG / "R0e_연령곡선.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

# ================================================================ R4 효율적 프론티어 추가 기간
def lp_max(mu, lb=0.01, ub=0.15):
    n = len(mu); w = np.full(n, lb); rest = 1 - n * lb
    for i in np.argsort(-mu):
        add = min(ub - lb, rest); w[i] += add; rest -= add
        if rest <= 1e-12:
            break
    return w


r4 = []
for pk in PKS:
    U = Uni.build(pk)
    top = np.argsort(-U.df["hist_yield"].to_numpy(), kind="mergesort")[:20]
    hy = U.arr["hist"][top] / U.arr["p0"][top][:, None]
    X = roll12(hy).T                       # 겹치는 12개월 누적 지급수익률 관측치 × 20곡
    for cov_lab, cov in [("Ledoit-Wolf", _ledoit_wolf(X)), ("표본공분산", np.cov(X, rowvar=False))]:
        mu = X.mean(0)
        W = {"동일금액 배분": np.full(20, 0.05), "최소분산 구성": _qp(cov, 0.01, 0.15),
             "동일 과거수익률에서 위험 최소화": _qp(cov, 0.01, 0.15, mu=mu, target=mu.mean() - 1e-12),
             "비중 상한 내 수익률 최대화": lp_max(mu)}
        for nm, w in W.items():
            S = top.reshape(1, -1)
            p = bundle_perf(U, S, w.reshape(1, -1))
            r4.append({"구성일": pk, "공분산": cov_lab, "구성방법": nm, "과거관측치수(겹치는12개월)": X.shape[0],
                       "과거 지급수익률(%)": float(w @ mu) * 100, "과거 추정위험(표준편차,%p)": float(np.sqrt(w @ cov @ w)) * 100,
                       "평가기간 실제 지급수익률(%)": p["yield"][0] * 100, "평가기간 실제 지급수익률_연환산(%)": p["yield"][0] * ANN[pk] * 100,
                       "평가기간 지급감소율(%)": p["dec_ltm"][0] * 100, "최대비중(%)": w.max() * 100})
R4 = pd.DataFrame(r4).round(3)
R4.to_csv(TB / "R4_프론티어_추가기간.csv", index=False, encoding="utf-8-sig")
print(R0e.to_markdown(index=False)); print(REG.round(4).to_markdown(index=False)); print("turning point", round(turn, 2))
print(R4[R4["공분산"] == "표본공분산"].drop(columns="공분산").to_markdown(index=False))
