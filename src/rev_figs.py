"""[수정 보완] 그림 R2: 특성별 짝 차이(Δ지급감소율) 분포, 기간별."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures import INK, PH_COLOR  # 한글 폰트·스타일
from config import CACHE
from rev_common import REV

warnings.filterwarnings("ignore")
D = pd.read_pickle(CACHE / "rev_pair_dist.pkl")
names = list(dict.fromkeys(D["전략"]))
LAB = {"P1": "2024 (선택)", "P2": "2025 (검증)", "P2b": "2026.1~8 (검증)"}
fig, ax = plt.subplots(figsize=(10, 9))
y = np.arange(len(names))[::-1]; h = 0.26
for i, pk in enumerate(["P1", "P2", "P2b"]):
    data = [D[(D["전략"] == nm) & (D["기간"] == pk)]["Δ"].to_numpy() for nm in names]
    bp = ax.boxplot(data, positions=y + (1 - i) * h, widths=h * 0.85, vert=False, whis=(5, 95), showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, lw=1.2))
    for b in bp["boxes"]:
        b.set_facecolor(PH_COLOR[pk]); b.set_alpha(0.85); b.set_edgecolor(PH_COLOR[pk])
    ax.scatter([d.mean() for d in data], y + (1 - i) * h, marker="D", s=14, color="#fcfcfb", edgecolor=INK, zorder=3)
    ax.plot([], [], color=PH_COLOR[pk], lw=6, label=LAB[pk])
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Δ 지급감소율 = 특성 추가 집합 - 같은 반복의 첫 집합 (%p) ← 음수가 개선")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
fig.suptitle("그림 R2. 특성 추가 구성과 재무정보(또는 기준) 구성의 짝 차이 분포 [수정보완]", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.text(0.01, -0.03, "주: 반복 3,000개의 짝 차이. 상자 = 25~75백분위, 수염 = 5~95백분위, ◇ = 평균. 3,000개는 같은 해 환경을 공유하는 비독립 표본이며, "
         "유의성은 아티스트 군집 부트스트랩(표 R2)으로만 판단한다.", fontsize=7.5, color="#5f5e5a")
fig.tight_layout()
fig.savefig(REV / "figures" / "R2_짝차이_분포.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print("saved")
