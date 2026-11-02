"""[수정 보완] 0단계: 원고 표 8(2025년, 20곡 동일금액, 규칙별 3,000개 평균) 재현.

원고 분석 코드(참고문헌 [19] 재현코드)는 저장소·zip에 없으므로 원고 정의로 재구현한다.
"""
import warnings
import numpy as np
import pandas as pd
from rev_common import (REV, N, POOL, K_DEFAULT, AGE_BINS, AGE_LABELS, elig, fin_score, top_pool, rand_sets,
                        paired_diversify, metrics, summ, _rng)

warnings.filterwarnings("ignore")
PK = "P2"
P, s = elig(PK)
s["age_bin3"] = pd.cut(s["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
MS = {"무작위": (6.22, 77.47, 9.49), "재무정보 기반": (9.81, 58.17, 5.05), "+ 연령 다양화": (9.75, 65.90, 6.11),
      "+ 수익원 분산": (9.79, 55.33, 4.66), "+ 아티스트 분산": (9.86, 56.50, 4.72), "+ 장르 분산": (10.10, 46.27, 3.35),
      "과거 감소 작은 곡": (8.35, 44.60, 3.62), "과거 수익률 중심": (9.51, 99.80, 19.97)}
fpool = top_pool(fin_score(s))
# 과거 감소 작은 곡: 과거 지급감소율 오름차순 상위 60곡. 동률(감소율 0인 곡 다수)은 무작위 순서로 끊는다(사전 가정).
rng = _rng("R0|S2|tie")
tie = pd.Series(rng.permutation(len(s)), index=s.index)
dpool = s.assign(_t=tie).sort_values(["hist_decline", "_t"]).index[:POOL]
ypool = top_pool(s["hist_yield"], ascending=False)
sets = {"무작위": rand_sets(s.index, "R0|S0"), "과거 수익률 중심": rand_sets(ypool, "R0|S1"),
        "과거 감소 작은 곡": rand_sets(dpool, "R0|S2")}
cats = {"+ 연령 다양화": s["age_bin3"], "+ 수익원 분산": s["main_source"], "+ 아티스트 분산": s["singer"]}
for nm, cat in cats.items():
    b, d = paired_diversify(fpool, f"R0|S3div|{nm}", cat)
    sets["재무정보 기반"] = b  # 모든 다양화 규칙에서 같은 태그 체계: 기준 = 각 반복의 첫 집합
    sets[nm] = d
# 재무정보 기반은 다양화와 짝을 이루는 첫 집합이므로, 다양화별로 기준이 달라지지 않게 공통 난수열을 쓴다.
base_common, _ = paired_diversify(fpool, "R0|S3div|common", s["singer"])
sets["재무정보 기반"] = base_common
for nm, cat in cats.items():
    b, d = paired_diversify(fpool, "R0|S3div|common", cat)   # 같은 난수열 → 같은 첫 집합
    assert (b == base_common).all()
    sets[nm] = d
rows = []
for nm, (my, mo, md) in MS.items():
    if nm not in sets:
        rows.append({"규칙": nm, "원고_지급수익률(%)": my, "원고_감소발생비중(%)": mo, "원고_평균감소율(%p)": md,
                     "재현_지급수익률(%)": np.nan, "재현_감소발생비중(%)": np.nan, "재현_평균감소율(%p)": np.nan,
                     "비고": "장르 분류자료가 저장소에 없어 재현 불가(후행정보, 탐색적)"}); continue
    r = summ(metrics(P, sets[nm]))
    ry, ro, rd = r["평균지급수익률"] * 100, r["지급감소발생비율"] * 100, r["평균지급감소율"] * 100
    rows.append({"규칙": nm, "원고_지급수익률(%)": my, "원고_감소발생비중(%)": mo, "원고_평균감소율(%p)": md,
                 "재현_지급수익률(%)": ry, "재현_감소발생비중(%)": ro, "재현_평균감소율(%p)": rd,
                 "차이_지급수익률(%p)": ry - my, "차이_감소발생비중(%p)": ro - mo, "차이_평균감소율(%p)": rd - md})
R0 = pd.DataFrame(rows)
R0["0.5%p초과칸수"] = (R0[["차이_지급수익률(%p)", "차이_감소발생비중(%p)", "차이_평균감소율(%p)"]].abs() > 0.5).sum(1)
R0 = R0.round(2)
R0.insert(0, "표시", "수정보완 0단계 재현")
R0.to_csv(REV / "tables" / "R0_표8_재현.csv", index=False, encoding="utf-8-sig")
# 표 9 대응: 재무정보 기반 대비 평균 감소율 변화
base_d = R0.set_index("규칙").loc["재무정보 기반", "재현_평균감소율(%p)"]
t9 = pd.DataFrame([{"추가 정보": k.replace("+ ", "").replace(" 다양화", "").replace(" 분산", ""),
                    "원고(%p)": {"+ 연령 다양화": 1.06, "+ 수익원 분산": -0.39, "+ 아티스트 분산": -0.33}[k],
                    "재현(%p)": R0.set_index("규칙").loc[k, "재현_평균감소율(%p)"] - base_d} for k in cats]).round(2)
t9.to_csv(REV / "tables" / "R0a_표9_재현.csv", index=False, encoding="utf-8-sig")
print(R0.drop(columns="표시").to_markdown(index=False)); print(); print(t9.to_markdown(index=False))
print("pools:", len(s), "fin top60 overlap with yield top60:", len(set(fpool) & set(ypool)), "| decline-pool ties at 0:", int((s.hist_decline == 0).sum()))
