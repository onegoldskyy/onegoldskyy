"""[수정 보완] 0.6단계 단독 통과 특성의 2개 조합(사전등록 규칙): M1(필터) + M3(다양화) @B-a.

rev_run.py의 조합 코드는 필터끼리만 다뤘으므로, 필터+다양화 조합을 같은 판정 기준으로 따로 계산한다.
"""
import warnings
import numpy as np
import pandas as pd
from rev_common import REV
from rev_fast import summarize_perf
from rev_strats import strategies_06, run_one, bootstrap, tercile_keep, _div, dollar_age
from rev_features import FEATS
import rev_run_helpers as H

warnings.filterwarnings("ignore")
S06 = strategies_06()
col, bad = FEATS["M1"][0], FEATS["M1"][1]


def m1m3(U, g, R):
    pos = np.flatnonzero(tercile_keep(U.df[col], bad))
    b, d = _div(U, pos, g, R, score_fn=dollar_age, maximize=True)
    return {"S": d}


rows = []
ref_fn = S06["B-a 무작위"][0]
for pk in ["P1", "P2", "P2b"]:
    U = H.build06(pk)
    _, p, _ = run_one(U, m1m3, f"combo|M1+M3|B-a|{pk}")
    _, r, _ = run_one(U, ref_fn, f"S06|B-a 무작위|{pk}")
    s, rb = summarize_perf(p, pk), summarize_perf(r, pk)
    row = {"조합": "M1+M3@B-a", "기간": pk, "평균감소율(%)": s["평균감소율"] * 100, "기준(B-a)_평균감소율(%)": rb["평균감소율"] * 100,
           "Δ감소율(%p)": (s["평균감소율"] - rb["평균감소율"]) * 100,
           "Δ수익률_연환산(%p)": (s["지급수익률_연환산"] - rb["지급수익률_연환산"]) * 100}
    if pk != "P1":
        bd = bootstrap(U, m1m3, ref_fn, f"combo|M1+M3|{pk}", 200, 500)
        row["부트95%_하한"], row["부트95%_상한"] = np.percentile(bd["d_dec"] * 100, [2.5, 97.5])
    rows.append(row)
C = pd.DataFrame(rows)
c1 = C.loc[C["기간"] == "P1", "Δ감소율(%p)"].iloc[0] < 0
c2 = (C.loc[C["기간"] != "P1", "Δ감소율(%p)"] < 0).all()
c3 = (C["Δ수익률_연환산(%p)"] >= -0.3).all()
c4 = (C["부트95%_상한"].dropna() < 0).any()
C["판정(조건1~4)"] = "재무가치 있음" if (c1 and c2 and c3 and c4) else "일관되지 않음"
C["비고"] = "2개 조합(사전등록: 단독 통과 특성끼리). Holm 보정 대상 밖이므로 조건 1~4로 판정"
C.round(3).to_csv(REV / "tables" / "R0c2_통과특성_2개조합.csv", index=False, encoding="utf-8-sig")
print(C.round(3).to_markdown(index=False))
