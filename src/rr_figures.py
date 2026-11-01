"""[보조·사후 탐색] 추가작업 그림 4종 (PNG 300dpi, 기존 그림과 같은 폰트·색 규칙)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures import BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, PH_LABEL, PH_COLOR  # 폰트·스타일 설정 포함
from config import CACHE
from rr_events import EXTRA

TB, FG = EXTRA / "tables", EXTRA / "figures"
NOTE = "보조·사후 탐색"


def read(name):
    return pd.read_csv(TB / f"{name}.csv", encoding="utf-8-sig")


def save(fig, name):
    fig.savefig(FG / f"{name}.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_path():
    P = pd.read_pickle(CACHE / "rr_paths.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, t in [(axes[0], "ratio_3m", "최근 3개월 평균 기준 (주지표)"), (axes[1], "ratio_1m", "월별 값 기준")]:
        g = P.groupby("k")[col]
        med, q1, q3, n = g.median(), g.quantile(.25), g.quantile(.75), g.count()
        k = med.index
        ax.fill_between(k, q1, q3, color=BLUE, alpha=0.18, lw=0, label="사분위 범위(25~75%)")
        ax.plot(k, med, color=BLUE, lw=2, label="중앙값")
        ax.axhline(1, color=INK, lw=1, ls="--"); ax.text(-12, 1.05, "사건 전 수준 = 1", fontsize=7.5, color=INK, va="bottom")
        ax.axvline(0, color="#898781", lw=1, ls=":")
        for kk in [3, 6, 12, 24]:
            if kk in med.index:
                ax.scatter(kk, med[kk], s=30, color=BLUE, edgecolor="#fcfcfb", zorder=3)
                ax.text(kk + 0.5, med[kk] * 1.15, f"{med[kk]:.1f}배\n(n={n[kk]})", ha="left", va="bottom", fontsize=7.5, color=INK)
        ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8]); ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("사건 시작월 기준 개월(0 = 시작월)"); ax.set_title(t, loc="left")
    axes[0].set_ylabel("전송 지급 ÷ 사건 전 12개월 평균 (로그 축)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"그림 3-7. 역주행 전후 전송 지급 경로 [{NOTE}]", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 역주행 = 발매 3년 이상 곡의 전송 지급이 직전 12개월 평균의 3배 이상인 달이 2개월 연속(32건). "
             "자료가 2026-08에서 끝나 늦은 사건은 +12·+24개월 값이 없다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout(); save(fig, "3-7_역주행_전후경로")


def fig_freq():
    F = read("3-11_역주행_빈도")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 5]})
    for ax, grp, t in [(axes[0], "사건시 곡연령", "사건 시 곡 연령"), (axes[1], "발매연대", "발매 연대")]:
        g = F[F["구분"] == grp]
        x = np.arange(len(g))
        ax.bar(x, g["100곡년당사건수"], color=BLUE, width=0.6, edgecolor="#fcfcfb", linewidth=2)
        for xi, v, e, s in zip(x, g["100곡년당사건수"], g["사건수"], g["판정가능곡수"]):
            ax.text(xi, v, f"{v:.2f}\n({e}건/{s}곡)", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(g["범주"], fontsize=8); ax.set_title(t, loc="left")
        ax.grid(axis="x", visible=False); ax.set_ylim(0, g["100곡년당사건수"].max() * 1.35)
    axes[0].set_ylabel("100곡·년당 사건 수")
    fig.suptitle(f"그림 3-8. 곡 연령·발매 연대별 역주행 빈도 [{NOTE}]", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.04, "주: 분모 = 판정 가능 곡-월 ÷ 12. 1990년대 이전은 7곡(사건 2건)으로 표본이 매우 작다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout(); save(fig, "3-8_역주행_빈도")


def fig_hold():
    C = read("3-13_역주행_묶음기여")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(C)); wd = 0.36
    a = C["단일곡_사건곡보유확률(%)"]; b = C["묶음_1곡이상보유비율(%,모의)"]
    ax.bar(x - wd / 2, a, wd, color=AQUA, label="단일곡 1개 보유: 사건 곡일 확률", edgecolor="#fcfcfb", linewidth=2)
    ax.bar(x + wd / 2, b, wd, color=BLUE, label="20곡 묶음: 사건 곡 1곡 이상 포함 확률", edgecolor="#fcfcfb", linewidth=2)
    for xi, v in list(zip(x - wd / 2, a)) + list(zip(x + wd / 2, b)):
        ax.text(xi, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PH_LABEL[p]}\n(후보 {n}곡 중 사건 {k}곡)" for p, n, k in
                        zip(C["기간"], C["적격후보곡수(A3)"], C["후보중_평가기간사건곡수"])], fontsize=8)
    ax.set_ylabel("확률 (%)"); ax.set_ylim(0, max(b.max(), 5) * 1.3); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"그림 3-9. 단일곡 vs 묶음의 역주행 곡 보유 확률 [{NOTE}]", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.06, "주: 최종 전략(A3 적격, 20곡, 동일 금액) 묶음 3,000개. 사건 곡 = 사건 시작월이 평가기간 안에 있는 곡. "
             "2024년은 후보 안에 사건 곡이 없었다.", fontsize=7.5, color="#5f5e5a")
    fig.tight_layout(); save(fig, "3-9_역주행_보유확률")


def fig_theme():
    D = pd.read_pickle(CACHE / "theme_diff.pkl")
    V = read("3-15b_테마포장_판정").set_index("테마")
    themes = list(dict.fromkeys(D["테마"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(themes))[::-1]; h = 0.26
    for ax, col, th, t in [(axes[0], "Δ평균지급감소율", 1.0, "평균 지급감소율 차이 (%p) ← 음수가 좋음"),
                           (axes[1], "Δ평균지급수익률_연환산", -0.3, "지급수익률 차이 (%p, 2026 연환산) → 양수가 좋음")]:
        for i, pk in enumerate(["P1", "P2", "P2b"]):
            vals = [D[(D["테마"] == tm) & (D["기간"] == pk)][col].iloc[0] * 100 for tm in themes]
            ax.barh(y + (1 - i) * h, vals, h, color=PH_COLOR[pk], label=PH_LABEL[pk], edgecolor="#fcfcfb", linewidth=1.5)
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(th, color=ORANGE if False else "#898781", lw=1.2, ls="--")
        ax.text(th, y.max() + 0.55, f" 판정 기준 {th:+.1f}%p", fontsize=7.5, color=INK)
        ax.set_title(t, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{tm}  [{V.loc[tm, '판정']}]" for tm in themes], fontsize=8)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"그림 3-10. 테마 포장의 T0(기본형) 대비 차이, 세 기간 [{NOTE}]", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.03, "주: 모두 A3 적격(T4만 A1), 20곡, 동일 금액. [작다] = 세 기간 모두 감소율 악화 ≤ 1%p이고 수익률 하락 ≤ 0.3%p(사전 고정 기준).",
             fontsize=7.5, color="#5f5e5a")
    fig.tight_layout(); save(fig, "3-10_테마포장_T0대비")


def main():
    fig_path(); fig_freq(); fig_hold(); fig_theme()
    print(sorted(p.name for p in FG.glob("*.png")))


if __name__ == "__main__":
    main()
