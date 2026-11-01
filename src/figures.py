"""5단계: 그림 (PNG 300dpi, 한글 폰트). 표(CSV)와 캐시에서 읽어 그린다."""
import json
import pickle
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager
from config import TABLES, FIGURES, CACHE, RESULTS
from engine import run

warnings.filterwarnings("ignore")

# ---- 한글 폰트: 맥 AppleGothic, 윈도우 Malgun Gothic, 그 외 NanumGothic
_avail = {f.name for f in font_manager.fontManager.ttflist}
for _f in {"Darwin": ["AppleGothic"], "Windows": ["Malgun Gothic"]}.get(platform.system(), []) + \
          ["NanumGothic", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#898781",
                     "axes.labelcolor": "#3d3c38", "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b5b3ad"
INK = "#3d3c38"
PH_LABEL = {"P1": "2024 (Phase 1)", "P2": "2025 (Phase 2)", "P2b": "2026.1~8 (Phase 2b)"}
PH_COLOR = {"P1": BLUE, "P2": ORANGE, "P2b": AQUA}
fz = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
FINAL = fz["final"]["spec"]


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv", encoding="utf-8-sig")


def short(s):
    return s.replace(" / C0 / N=20 / E1", "").replace("A3(|LTM/LTY−1|≤0.2)", "A3(≤0.2)").replace("|LTM/LTY−1|", "안정도")


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- 1. 단계별 후보 비교
def fig_steps():
    steps = [("A. 적격 기준", "3-3a_PhaseA_적격기준비교", lambda s: s.split(" / ")[0]),
             ("B. 묶는 축", "3-3b_PhaseB_묶는축비교", lambda s: s.split(" / ")[1]),
             ("C. 분산 규칙", "3-3c_PhaseC_분산규칙비교", lambda s: s.split(" / ")[2]),
             ("E. 편입 비중", "3-3e_PhaseE_편입비중", lambda s: s.split(" / ")[4])]
    fig, axes = plt.subplots(4, 2, figsize=(10, 12), gridspec_kw={"width_ratios": [1, 1]})
    for r, (title, tab, lab) in enumerate(steps):
        d = read(tab)
        d = d[d["실행가능"] == True].sort_values("순위")
        names = [lab(s).replace("|LTM/LTY−1|", "안정도") for s in d["전략"]]
        y = np.arange(len(d))[::-1]
        col = [BLUE if s else GRAY for s in d["선택"]]
        for c, (metric, xl) in enumerate([("평균지급감소율(%)", "평균 지급감소율 (%) ← 낮을수록 좋음"),
                                          ("평균지급수익률(%)", "평균 지급수익률 (%)")]):
            ax = axes[r, c]
            ax.barh(y, d[metric], color=col, height=0.62, edgecolor="#fcfcfb", linewidth=2)
            for yi, v in zip(y, d[metric]):
                ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8, color=INK)
            if c == 1:
                ax.axvline(d["비교B0수익률(%)"].iloc[0], color=ORANGE, lw=1.5, ls="--")
                ax.text(d["비교B0수익률(%)"].iloc[0], y.max() + 0.6, " 비교 B0 수익률", color=INK, fontsize=7.5)
                ax.set_yticks(y); ax.set_yticklabels([""] * len(y))
            else:
                ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
                ax.set_title(f"{title}  (파랑 = 선택)", loc="left")
            ax.set_xlabel(xl)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, d[metric].max() * 1.22)
    fig.suptitle("그림 3-1. Phase 1(2024) 단계별 후보 비교 (가상 묶음 3,000개 평균)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.005, "주: 판단 기준 = 평균 지급감소율 최저, 평균 지급수익률 ≥ 비교 B0, 차이 0.5%p 미만이면 단순한 쪽, 동률이면 감소율 p95.",
             fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-1_단계별_후보비교")


# ---------------------------------------------------------------- 2. 곡 수 곡선
def fig_n_curve():
    rows = []
    for pk in ["P1", "P2", "P2b"]:
        for n in [5, 10, 20, 30]:
            r = run(pk, {**FINAL, "N": n})
            rows.append({"기간": pk, "N": n, "평균감소율": r["평균지급감소율"] * 100, "p95": r["감소율_p95"] * 100})
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "3-3d2_곡수곡선_기간별.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, t in [(axes[0], "평균감소율", "평균 지급감소율 (%)"), (axes[1], "p95", "지급감소율 95백분위 (%)")]:
        for pk in ["P1", "P2", "P2b"]:
            g = d[d["기간"] == pk]
            ax.plot(g["N"], g[m], color=PH_COLOR[pk], lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=PH_LABEL[pk])
            ax.text(31, g[m].iloc[-1], f" {g[m].iloc[-1]:.1f}", va="center", fontsize=8, color=INK)
        ax.axvline(20, color="#898781", lw=1, ls=":")
        lo_, hi_ = ax.get_ylim()
        ax.text(20, lo_ + (hi_ - lo_) * 0.03, " 선택: 20곡", fontsize=8, color=INK, va="bottom")
        ax.set_xticks([5, 10, 20, 30]); ax.set_xlabel("묶음 곡 수 (N)"); ax.set_title(t, loc="left")
        ax.set_xlim(3, 34)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("그림 3-2. 곡 수에 따른 지급감소율 (최종 전략의 적격 기준·동일 금액)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 곡 수 선택은 Phase 1(2024)에서만 했다. 2025·2026 곡선은 같은 규칙을 적용한 검증용이다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-2_곡수곡선")


# ---------------------------------------------------------------- 3. 분포 박스플롯
def fig_box():
    raw = pickle.load(open(CACHE / "phase2_bundle_metrics.pkl", "rb"))
    groups = [("단일곡(최종 적격기준)", "단일곡", AQUA), ("B0무작위(A0)", "무작위 20곡", ORANGE), ("최종", "최종 전략", BLUE)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey="row")
    for c, pk in enumerate(["P1", "P2", "P2b"]):
        for r, (m, yl) in enumerate([("decline", "지급감소율 (%)"), ("yield", "지급수익률 (%)")]):
            ax = axes[r, c]
            data = [raw[(pk, k)][m].dropna().to_numpy() * 100 for k, _, _ in groups]
            bp = ax.boxplot(data, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.5))
            for patch, (_, _, col) in zip(bp["boxes"], groups):
                patch.set_facecolor(col); patch.set_alpha(0.85); patch.set_edgecolor(col)
            for i, v in enumerate(data):
                ax.scatter(i + 1, v.mean(), marker="D", s=22, color="#fcfcfb", edgecolor=INK, zorder=3)
                ax.text(i + 1.32, v.mean(), f"{v.mean():.1f}", fontsize=7.5, color=INK, va="center")
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels([g[1] for g in groups], fontsize=8)
            if c == 0:
                ax.set_ylabel(yl)
            if r == 0:
                ax.set_title(PH_LABEL[pk], loc="left")
    fig.suptitle("그림 3-3. 최종 전략 vs 무작위 vs 단일곡 분포", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.01, "주: 상자 = 25~75백분위, 수염 = 5~95백분위, ◇ = 평균(숫자). 단일곡은 최종 전략과 같은 적격 기준의 곡 하나 보유. "
             "2026은 8개월 지급. 묶음 3,000개는 같은 해 환경을 공유하는 비독립 표본이다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-3_분포_박스플롯")


# ---------------------------------------------------------------- 4. 연대별 묶음 성과
def fig_decade():
    rows = []
    from engine import PhaseData, apply_A
    for pk in ["P1", "P2", "P2b"]:
        P = PhaseData(pk)
        el = apply_A(P.sel, FINAL["A"])
        sup = P.sel.loc[el, "decade"].value_counts()
        for dec, n in sup.items():
            r = run(pk, {**FINAL, "B": {"code": "B1", "decade": dec}})
            rows.append({"기간": pk, "연대": dec, "공급곡수": n, "실행가능(공급≥60)": r["실행가능"],
                         "평균지급감소율(%)": r.get("평균지급감소율", np.nan) * 100 if r["실행가능"] else np.nan,
                         "평균지급수익률(%)": r.get("평균지급수익률", np.nan) * 100 if r["실행가능"] else np.nan})
        b = run(pk, FINAL)
        rows.append({"기간": pk, "연대": "전체(최종 전략)", "공급곡수": len(el), "실행가능(공급≥60)": True,
                     "평균지급감소율(%)": b["평균지급감소율"] * 100, "평균지급수익률(%)": b["평균지급수익률"] * 100})
    d = pd.DataFrame(rows).round(2)
    d.to_csv(TABLES / "3-3b2_연대별_묶음성과_기간별.csv", index=False, encoding="utf-8-sig")
    ok = d[d["실행가능(공급≥60)"]]
    cats = [c for c in ["2000년대", "2010년대", "전체(최종 전략)"] if c in set(ok["연대"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(cats)); wd = 0.26
    for ax, m in [(axes[0], "평균지급감소율(%)"), (axes[1], "평균지급수익률(%)")]:
        for i, pk in enumerate(["P1", "P2", "P2b"]):
            vals = [ok[(ok["기간"] == pk) & (ok["연대"] == c)][m].mean() for c in cats]
            ax.bar(x + (i - 1) * wd, vals, wd, color=PH_COLOR[pk], label=PH_LABEL[pk], edgecolor="#fcfcfb", linewidth=2)
            for xi, v in zip(x + (i - 1) * wd, vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_title(m.replace("(%)", " (%)"), loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("그림 3-4. 발매 연대별 묶음 성과 (최종 적격 기준, 20곡, 동일 금액)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 공급 곡 수가 60곡(3N) 이상인 연대만 표시. 1990년대 이전·1990년대·2020년대는 공급 부족. "
             "B1은 Phase 1 규칙(0.5%p)으로 탈락했으며 2025·2026 결과는 사후 탐색이다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-4_연대별_묶음성과")


# ---------------------------------------------------------------- 5. 연령 구간별 가격배수
def fig_multiple():
    R = pd.read_pickle(CACHE / "multiple_panel.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, col, order, t in [(axes[0], "본연구연령구간", ["<7년", "7~15년", "15년+"], "본 연구 연령 구간"),
                              (axes[1], "RE연령구간", ["0~3년", "3~10년", "10년+"], "Royalty Exchange 연령 구간")]:
        data = [R.loc[R[col].astype(str) == k, "multiple"].to_numpy() for k in order]
        bp = ax.boxplot(data, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.5))
        for p in bp["boxes"]:
            p.set_facecolor(BLUE); p.set_alpha(0.8); p.set_edgecolor(BLUE)
        for i, v in enumerate(data):
            ax.text(i + 1.32, np.median(v), f"{np.median(v):.1f}배\n(n={len(v)})", fontsize=7.5, color=INK, va="center")
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(order); ax.set_title(t, loc="left")
        ax.set_yscale("log"); ax.set_yticks([5, 10, 20, 40, 80]); ax.set_yticklabels(["5", "10", "20", "40", "80"])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("가격배수 = 기초가격 ÷ LTM (로그 축)")
    fig.suptitle("그림 3-5. 곡 연령 구간별 가격배수 (2024·2025·2026 구성일 적격 곡 합산)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 상자 = 25~75백분위, 수염 = 5~95백분위, 숫자 = 중앙값. 1주 가격 ÷ 1주 연간 지급이므로 Royalty Exchange 경매 배수(0~3년 약 4배, "
             "3~10년 5~7배, 10년+ 8배 이상)와 수준을 직접 비교할 수 없다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-5_연령구간별_가격배수")


# ---------------------------------------------------------------- 6. 바스켓 체결 가능성
def fig_basket():
    m = read("3-6b_월말별_체결비율")
    b = read("3-6_바스켓_체결가능성")
    m["월말"] = pd.to_datetime(m["월말"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for col, c, lab in [("30일내체결비율(%)", AQUA, "최근 30일 내 체결"), ("7일내체결비율(%)", BLUE, "최근 7일 내 체결")]:
        ax.plot(m["월말"], m[col], color=c, lw=2, label=lab)
        ax.text(m["월말"].iloc[-1], m[col].iloc[-1], f" {m[col].iloc[-1]:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_ylim(0, 100); ax.set_ylabel("cohort 767곡 중 비율 (%)"); ax.set_title("월말 기준 체결 곡 비율", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    x = np.arange(len(b)); wd = 0.36
    v1 = b["최종묶음_20곡모두7일내체결확률(%)"]; v2 = b["가격무조건_20곡모두30일내확률(%)"]
    ax.bar(x - wd / 2, v1, wd, color=BLUE, label="최종 전략 20곡 모두 7일 내", edgecolor="#fcfcfb", linewidth=2)
    ax.bar(x + wd / 2, v2, wd, color=AQUA, label="가격조건 없는 후보 20곡 모두 30일 내", edgecolor="#fcfcfb", linewidth=2)
    for xi, v in list(zip(x - wd / 2, v1)) + list(zip(x + wd / 2, v2)):
        ax.text(xi, v + 0.1, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in b["구성기준일"]], fontsize=8)
    ax.set_ylim(0, max(5, float(v2.max()) * 1.6)); ax.set_ylabel("확률 (%)")
    ax.set_title("20곡을 같은 시점 가격으로 살 수 있었을 확률", loc="left"); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.suptitle("그림 3-6. 가격 번들(바스켓) 체결 가능성", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 체결 = traded_daily_close·거래량>0 조건을 만족한 체결 종가가 기준일 이전 7(30)일 내 존재. 확률은 무작위 20곡 3,000회 추출 기준.",
             fontsize=7.5, color="#5f5e5a")
    fig.tight_layout()
    save(fig, "3-6_바스켓_체결가능성")


def main():
    fig_steps(); fig_n_curve(); fig_box(); fig_decade(); fig_multiple(); fig_basket()
    print("font:", plt.rcParams["font.family"], "| figures:", sorted(p.name for p in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()
