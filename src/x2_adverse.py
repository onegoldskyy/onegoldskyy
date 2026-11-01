"""[보조 분석] 5단계: 역선택 스트레스 — 구성 시점 정보만으로 투자자에게 불리하게 고른 20곡 묶음."""
import warnings
import numpy as np
import pandas as pd
from engine import run, apply_A, PhaseData, bundle_metrics, summarize, sample_bundles, weights
from x2_common import TAG, LG, FINAL, A0, tsave

warnings.filterwarnings("ignore")
N = 20
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}
KEYS = [("평균지급수익률", "지급수익률"), ("평균지급감소율", "평균감소율"), ("감소율_p95", "감소율_p95"),
        ("총수익률_평균", "총수익률_평균"), ("총수익률_p5", "총수익률_p5")]


def one(P, ids):
    m = bundle_metrics(P, np.asarray(ids).reshape(1, -1), np.full((1, len(ids)), 1 / len(ids)))
    return {"지급수익률": m["yield"].iloc[0], "평균감소율": m["decline"].iloc[0], "감소율_p95": np.nan,
            "총수익률_평균": m["total_return"].iloc[0], "총수익률_p5": np.nan, "가격커버리지": m["coverage"].iloc[0]}


rows, overlap = [], []
for pk in ["P1", "P2", "P2b"]:
    P = PhaseData(pk); s = P.sel
    base = run(pk, FINAL)
    rows.append({"기간": pk, "적격기준": "A3", "구분": "최종 묶음(무작위, 3,000개 평균)", "묶음수": 3000,
                 **{b: base[a] for a, b in KEYS}, "가격커버리지": base["가격커버리지_평균"]})
    for lab, A in [("A3", FINAL["A"]), ("A0(참고: 적격 기준 없음)", A0)]:
        el = s.loc[apply_A(s, A)]
        el = el[el["ltm"] > 0]
        s1 = el["multiple"].sort_values(ascending=False, kind="mergesort").index[:N]
        s2 = el["hist_yield"].sort_values(ascending=True, kind="mergesort").index[:N]
        s3 = el["ltm_lty"].sort_values(ascending=True, kind="mergesort").index[:N]
        top60 = el["multiple"].sort_values(ascending=False, kind="mergesort").index[:3 * N]
        overlap.append({"기간": pk, "적격기준": lab, "S1∩S2 곡수": len(set(s1) & set(s2)), "S1∩S3 곡수": len(set(s1) & set(s3)),
                        "S3 LTM/LTY 최솟값": el.loc[s3, "ltm_lty"].min(), "S3 LTM/LTY 최댓값": el.loc[s3, "ltm_lty"].max()})
        for nm, ids in [("S1 가격배수 최고 20곡", s1), ("S2 지급수익률 최저 20곡", s2), ("S3 LTM/LTY 최저 20곡", s3)]:
            rows.append({"기간": pk, "적격기준": lab, "구분": nm, "묶음수": 1, **one(P, ids)})
        spec = {"A": A, "B": {"code": "B0", "decade": None}, "C": "C0", "N": N, "E": "E1"}
        b, _ = sample_bundles(s, pd.Index(top60), spec, f"x2|S4|{lab}|{pk}")
        r = summarize(bundle_metrics(P, b, weights(s, b, "E1")))
        rows.append({"기간": pk, "적격기준": lab, "구분": "S4 배수 상위 60곡 중 무작위 20곡(3,000개)", "묶음수": 3000,
                     **{bb: r[a] for a, bb in KEYS}, "가격커버리지": r["가격커버리지_평균"]})
T = pd.DataFrame(rows)
# 최종 묶음 대비 차이
for pk in ["P1", "P2", "P2b"]:
    ref = T[(T["기간"] == pk) & T["구분"].str.startswith("최종")].iloc[0]
    m = T["기간"] == pk
    T.loc[m, "Δ지급수익률(%p)"] = (T.loc[m, "지급수익률"] - ref["지급수익률"]) * 100
    T.loc[m, "Δ평균감소율(%p)"] = (T.loc[m, "평균감소율"] - ref["평균감소율"]) * 100
    T.loc[m, "Δ총수익률평균(%p)"] = (T.loc[m, "총수익률_평균"] - ref["총수익률_평균"]) * 100
T["지급수익률_연환산"] = T["지급수익률"] * T["기간"].map(ANN)
pc = ["지급수익률", "지급수익률_연환산", "평균감소율", "감소율_p95", "총수익률_평균", "총수익률_p5", "가격커버리지"]
To = T.copy(); To[pc] = (To[pc] * 100).round(2); To = To.rename(columns={c: c + "(%)" for c in pc}).round(3)
To = To[["기간", "적격기준", "구분", "묶음수", "지급수익률(%)", "지급수익률_연환산(%)", "Δ지급수익률(%p)", "평균감소율(%)", "Δ평균감소율(%p)",
         "감소율_p95(%)", "총수익률_평균(%)", "Δ총수익률평균(%p)", "총수익률_p5(%)", "가격커버리지(%)"]]
tsave(To, "3-26_역선택스트레스")
O = pd.DataFrame(overlap).round(3)
tsave(O, "3-26b_역선택_구성겹침")
# 판단 문장 (사전 형식): A3 안의 S1~S4 중 최악
sent = []
for pk in ["P1", "P2", "P2b"]:
    a3 = T[(T["기간"] == pk) & (T["적격기준"] == "A3") & T["구분"].str.startswith("S")]
    a0 = T[(T["기간"] == pk) & (T["적격기준"].str.startswith("A0")) & T["구분"].str.startswith("S")]
    wy, wd = a3.loc[a3["Δ지급수익률(%p)"].idxmin()], a3.loc[a3["Δ평균감소율(%p)"].idxmax()]
    sent.append({"기간": pk, "A3_최대수익률하락(%p)": -wy["Δ지급수익률(%p)"], "해당묶음": wy["구분"],
                 "A3_최대감소율상승(%p)": wd["Δ평균감소율(%p)"], "해당묶음(감소율)": wd["구분"],
                 "A0참고_최대수익률하락(%p)": -a0["Δ지급수익률(%p)"].min(), "A0참고_최대감소율상승(%p)": a0["Δ평균감소율(%p)"].max()})
Sx = pd.DataFrame(sent).round(2)
tsave(Sx, "3-26c_역선택_판단요약")
ymax, dmax = Sx["A3_최대수익률하락(%p)"].max(), Sx["A3_최대감소율상승(%p)"].max()
a3s = T[(T["적격기준"] == "A3") & T["구분"].str.startswith("S")]
trmax = -a3s["Δ총수익률평균(%p)"].min()
trwho = a3s.loc[a3s["Δ총수익률평균(%p)"].idxmin()]
dtxt = (f"평균 감소율이 최대 {dmax:.2f}%p 높아질 수 있다" if dmax > 0 else
        f"평균 감소율은 최대 {dmax:.2f}%p 높아지는 데 그쳐 실제로는 높아지지 않았다(세 기간 모두 최종 묶음보다 낮음)")
judg = (f"적격 기준은 체리피킹 피해의 상한을 제한하지만, 무작위 편입과 곡별 공시가 없으면 최종 묶음보다 지급수익률이 최대 {ymax:.2f}%p 낮고 "
        f"{dtxt}. 총수익률(평균)은 최대 {trmax:.2f}%p 낮아질 수 있다({trwho['기간']} {trwho['구분']}). (세 기간 중 최악, A3 안의 S1~S4 기준)")
log = [f"# [{TAG}] 보강 5단계: 역선택 스트레스\n",
       "- 모든 선정은 구성 시점 정보(sel)만 쓴다. S1: 가격배수(기초가격÷LTM) 최고 20곡, S2: 과거 지급수익률(LTM÷기초가격) 최저 20곡, "
       "S3: LTM/LTY 최저 20곡, S4: S1 기준 상위 60곡 중 무작위 20곡(3,000개). 모두 동일 금액.",
       "- S1과 S2는 정의상 역수 관계(배수 = 1/지급수익률)라 같은 20곡이 된다(표 3-26b). 둘 다 지시대로 보고한다.",
       "- S1~S3은 묶음 하나라 감소율 p95·총수익률 p5가 없다(빈칸). 참고로 적격 기준이 없을 때(A0)의 같은 선정도 계산해 적격 기준의 방어 효과를 비교했다.",
       To.to_markdown(index=False), "", O.to_markdown(index=False), "", Sx.to_markdown(index=False), "",
       f"- **판단(사전 형식): {judg}**",
       "- 해석: A3 안에서 비싼 곡(S1·S2)이나 지급이 줄던 곡(S3)을 골라도 다음 해 감소율은 오히려 낮았다. 구성 직전 지급(LTM)이 작은 곡은 다음 해 지급이 "
       "되돌아오는(평균 회귀) 경향이 있고, 감소율의 기준이 그 LTM이기 때문이다. 체리피킹의 피해는 감소율이 아니라 **낮은 지급수익률과 큰 가격 손실**로 나타난다.",
       "- S1~S3 단일 묶음 중 기말가격 커버리지가 80% 미만이면 기존 규칙대로 총수익률을 계산하지 않았다(빈칸).",
       f"- 참고(A0, 적격 기준 없음): 같은 방식의 최악은 수익률 최대 {Sx['A0참고_최대수익률하락(%p)'].max():.2f}%p 하락, 감소율 최대 "
       f"{Sx['A0참고_최대감소율상승(%p)'].max():.2f}%p 상승이다.\n"]
(LG / "24_역선택스트레스.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
