"""2단계: Phase 1(2024)에서 A→B→C→D→E를 한 단계씩 고정하고 frozen_strategy.json 저장.

평가 결과는 Phase 1(2024) 성과만 사용한다. Phase 2·2b 자료는 이 스크립트에서 읽지 않는다.
"""
import copy
import json
import warnings
import numpy as np
import pandas as pd
from config import RESULTS, LOGS, SEED, N_BUNDLES
from engine import run, choose, spec_name, apply_A, PhaseData, TOL
from report_util import save, md

warnings.filterwarnings("ignore")
PK = "P1"
log = ["# 2단계. Phase 1(2024) 단계별 선택\n",
       f"- seed={SEED}, 단계별 가상 묶음 {N_BUNDLES:,}개. 판단 기준(6절): ① 평균 지급감소율 최저 ② 평균 지급수익률 ≥ 같은 적격 기준 B0 무작위 "
       "③ 최저값과 차이 0.5%p 미만이면 단순한 쪽(규칙 수, E1 우선) ④ 동률이면 감소율 95백분위 낮은 쪽.",
       "- 묶음 지급감소율 = max(0, 1 − 묶음 평가기간 지급 ÷ 묶음 비교기간 지급). 묶음 지급은 투자금 비중대로 산 주식 수 × 곡별 1주당 지급이므로 투자금 가중이다. "
       "곡별 감소율의 투자금 가중 평균은 보조 열로 함께 보고한다.",
       "- 3,000개 묶음은 같은 해 환경을 공유하므로 독립 표본이 아니다. 분포는 '이 해에 이 규칙으로 만들 수 있었던 묶음의 범위'로만 해석한다.\n"]
A0 = {"code": "A0", "dec_th": None, "stab_th": None}


def mk(A=A0, B=None, C="C0", N=20, E="E1"):
    return {"A": copy.deepcopy(A), "B": copy.deepcopy(B or {"code": "B0", "decade": None}), "C": C, "N": N, "E": E}


def b0_yield(spec):
    s = mk(A=spec["A"], N=spec["N"])
    return run(PK, s)["평균지급수익률"]


def second(df, pick):
    r = df.sort_values("순위")
    r = r[(r.index != pick) & r["실행가능"].fillna(False).astype(bool)]
    return r.index[0] if len(r) else None


# ------------------------------------------------------------ A
A_cands = [A0, {"code": "A1", "dec_th": None, "stab_th": None}]
A_cands += [{"code": "A2", "dec_th": t, "stab_th": None} for t in (0.0, 0.05, 0.10)]
A_cands += [{"code": "A3", "dec_th": None, "stab_th": t} for t in (0.1, 0.2)]
A_cands += [{"code": "A1+A2", "dec_th": t, "stab_th": None} for t in (0.0, 0.05, 0.10)]
specsA = [mk(A=a) for a in A_cands]
rowsA = [run(PK, s) for s in specsA]
base_y = rowsA[0]["평균지급수익률"]
dfA, pA = choose(rowsA, base_y)
A_sel = specsA[pA]["A"]; A_2nd = specsA[second(dfA, pA)]["A"]
dA = save(dfA, "3-3a_PhaseA_적격기준비교")
log += ["## A. 적격 기준 (B0, C0, N=20, E1)\n",
        "- 해석: A 후보는 모두 B0이므로 조건②의 비교 기준은 A0 무작위(제한 없음)로 둔다.",
        md(dA), f"\n- **선택: {spec_name(specsA[pA]).split(' / ')[0]}**, 2등: {spec_name(specsA[second(dfA, pA)]).split(' / ')[0]}\n"]

# ------------------------------------------------------------ B
N = 20
P = PhaseData(PK)
elig = apply_A(P.sel, A_sel)
decades = P.sel.loc[elig, "decade"].value_counts()
yB0 = b0_yield(mk(A=A_sel))
rows_dec, specs_dec = [], []
for dec in sorted(decades.index):
    s = mk(A=A_sel, B={"code": "B1", "decade": dec})
    specs_dec.append(s); rows_dec.append(run(PK, s))
dfDec, pDec = choose(rows_dec, yB0)
dfDec["연대공급곡수"] = [decades[s["B"]["decade"]] for s in specs_dec]
dfDec["공급≥3N"] = dfDec["연대공급곡수"] >= 3 * N
dDec = save(dfDec, "3-3b1_PhaseB1_연대별")
best_dec_spec = specs_dec[pDec] if dfDec.loc[pDec, "공급≥3N"] else None
log += ["## B. 묶는 축 (A 고정, C0, N=20, E1)\n", "### B1 연대별 결과\n",
        md(dDec, ["순위", "선택", "전략", "연대공급곡수", "공급≥3N", "평균지급수익률(%)", "조건2_수익률≥B0", "평균지급감소율(%)", "감소율_p95(%)"]),
        f"\n- 공급 곡 수가 3N(=60) 미만인 연대는 '실행불가'로 처리했다. B1 대표: "
        f"{best_dec_spec['B']['decade'] if best_dec_spec else '없음'} (같은 판단 기준 적용).\n"]
specsB = [mk(A=A_sel)]
if best_dec_spec:
    specsB.append(best_dec_spec)
specsB += [mk(A=A_sel, B={"code": c, "decade": None}) for c in ("B2", "B3", "B4")]
rowsB = [run(PK, s) for s in specsB]
dfB, pB = choose(rowsB, yB0)
B_sel = specsB[pB]["B"]; B_2nd = specsB[second(dfB, pB)]["B"]
dB = save(dfB, "3-3b_PhaseB_묶는축비교")
log += ["### B 후보 비교\n", md(dB), f"\n- **선택: {spec_name(specsB[pB]).split(' / ')[1]}**, 2등: {spec_name(specsB[second(dfB, pB)]).split(' / ')[1]}\n"]

# ------------------------------------------------------------ C
specsC = [mk(A=A_sel, B=B_sel, C=c) for c in ("C0", "C1", "C2", "C1+C2")]
rowsC = [run(PK, s) for s in specsC]
dfC, pC = choose(rowsC, yB0)
C_sel = specsC[pC]["C"]; C_2nd = specsC[second(dfC, pC)]["C"]
dC = save(dfC, "3-3c_PhaseC_분산규칙비교")
heavy_n = int((P.sel.loc[elig, "broadcast_share"] >= 0.5).sum())
log += ["## C. 분산 규칙 (A·B 고정, N=20, E1)\n",
        f"- C1 상한은 ceil(N/10)곡(20곡이면 2곡). 적격 후보군에서 방송 비중 50% 이상인 곡은 {heavy_n}곡이다. C2는 동일 singer 문자열 최대 1곡이다.",
        md(dC, ["순위", "선택", "전략", "평균지급수익률(%)", "조건2_수익률≥B0", "평균지급감소율(%)", "최저감소율과차이(%p)", "복잡도", "감소율_p95(%)", "제약미충족묶음"]),
        f"\n- **선택: {C_sel}**, 2등: {C_2nd}\n"]

# ------------------------------------------------------------ D
Ns = [5, 10, 20, 30]
specsD = [mk(A=A_sel, B=B_sel, C=C_sel, N=n) for n in Ns]
rowsD = [run(PK, s) for s in specsD]
yB0_N = {spec_name(s): b0_yield(s) for s in specsD}
dfD = pd.DataFrame(rowsD)
dfD["비교B0수익률"] = dfD["전략"].map(yB0_N)
dfD["조건2_수익률≥B0"] = dfD["실행가능"].fillna(False).astype(bool) & (dfD["평균지급수익률"] >= dfD["비교B0수익률"] - 1e-12)
d20 = dfD.loc[Ns.index(20), "평균지급감소율"]
dfD["20곡대비차이(%p)"] = (dfD["평균지급감소율"] - d20) * 100
ok = dfD["조건2_수익률≥B0"]
# 기본안(계획에서 확인): 20곡과 감소율 차이가 0.5%p 미만인 최소 곡 수
rule = ok & ((dfD["평균지급감소율"] - d20).abs() < TOL)
pD = dfD.index[rule][0] if rule.any() else Ns.index(20)
# 대안: 전체 곡 수 중 최저 감소율과 0.5%p 이내인 최소 곡 수
best_all = dfD.loc[ok, "평균지급감소율"].min() if ok.any() else dfD["평균지급감소율"].min()
alt = ok & (dfD["평균지급감소율"] <= best_all + TOL - 1e-12)
pD_alt = dfD.index[alt][0] if alt.any() else pD
dfD["선택"] = dfD.index == pD
dfD["대안규칙선택"] = dfD.index == pD_alt
# 2등: 선택 외 곡 수 중 20곡과 가장 가까운(규칙 충족) 다음 후보
others = dfD[(dfD.index != pD) & dfD["실행가능"].fillna(False).astype(bool)]
d2 = (others.assign(_ok=~others["조건2_수익률≥B0"], _r=~((others["평균지급감소율"] - d20).abs() < TOL))
      .sort_values(["_ok", "_r", "평균지급감소율"]).index[0])
dfD["순위"] = 0
dfD.loc[pD, "순위"] = 1; dfD.loc[d2, "순위"] = 2
rest = [i for i in dfD.sort_values("평균지급감소율").index if i not in (pD, d2)]
for k, i in enumerate(rest, start=3):
    dfD.loc[i, "순위"] = k
N_sel, N_2nd = Ns[pD], Ns[d2]
dD = save(dfD, "3-3d_PhaseD_곡수")
log += ["## D. 곡 수 (A·B·C 고정, E1)\n",
        "- 규칙(기본안): 평균 지급감소율이 20곡과 0.5%p 미만 차이인 최소 곡 수. 대안: 4개 곡 수 중 최저 감소율과 0.5%p 이내인 최소 곡 수.",
        md(dD, ["순위", "선택", "대안규칙선택", "전략", "추출후보곡수", "평균지급수익률(%)", "비교B0수익률(%)", "조건2_수익률≥B0",
                "평균지급감소율(%)", "20곡대비차이(%p)", "감소율_p95(%)", "지급감소발생비율(%)"]),
        f"\n- **선택: N={N_sel}** (대안 규칙 결과 N={Ns[pD_alt]}{', 동일' if pD == pD_alt else ', 다름 → 기본안 유지'}), 2등: N={N_2nd}\n"]

# ------------------------------------------------------------ E
specsE = [mk(A=A_sel, B=B_sel, C=C_sel, N=N_sel, E=e) for e in ("E1", "E2", "E3", "E4")]
rowsE = [run(PK, s) for s in specsE]
yB0_E = b0_yield(specsE[0])
dfE, pE = choose(rowsE, yB0_E)
E_sel = specsE[pE]["E"]; E_2nd = specsE[second(dfE, pE)]["E"]
dE = save(dfE, "3-3e_PhaseE_편입비중")
log += ["## E. 편입 비중 (A·B·C·N 고정)\n",
        "- E2: w∝기초가격/LTM(곡별 기대 지급이 같도록). E3: w∝곡 전체 LTM(1주당 지급×주식 수). "
        "E4: 최근 36개월(가용분) 월별 지급수익률의 Ledoit-Wolf 축소 공분산으로 최소분산, 곡당 1~15%(N<7이면 상한 1/N).",
        md(dE, ["순위", "선택", "전략", "평균지급수익률(%)", "비교B0수익률(%)", "조건2_수익률≥B0", "평균지급감소율(%)",
                "최저감소율과차이(%p)", "복잡도", "감소율_p95(%)", "DollarAge_평균", "최대1곡지급비중_평균(%)", "최대1곡지급비중_p95(%)"]),
        f"\n- **선택: {E_sel}**, 2등: {E_2nd}\n"]

# ------------------------------------------------------------ 최종·차순위
final = mk(A=A_sel, B=B_sel, C=C_sel, N=N_sel, E=E_sel)
variants = {"A": mk(A=A_2nd, B=B_sel, C=C_sel, N=N_sel, E=E_sel),
            "B": mk(A=A_sel, B=B_2nd, C=C_sel, N=N_sel, E=E_sel),
            "C": mk(A=A_sel, B=B_sel, C=C_2nd, N=N_sel, E=E_sel),
            "D": mk(A=A_sel, B=B_sel, C=C_sel, N=N_2nd, E=E_sel),
            "E": mk(A=A_sel, B=B_sel, C=C_sel, N=N_sel, E=E_2nd)}
rowsV = [run(PK, s) for s in variants.values()]
yV = {spec_name(s): b0_yield(s) for s in variants.values()}
dfV, _ = choose(rowsV, yV)
dfV.insert(0, "바꾼단계", list(variants.keys()))
dfV["선택"] = False
top2 = dfV.sort_values("순위").index[:2]
dV = save(dfV, "3-3f_차순위후보")
runner = [list(variants.values())[i] for i in top2]
final_row = run(PK, final)
log += ["## 최종 전략과 차순위\n",
        f"- 최종: **{spec_name(final)}** (Phase 1 평균 지급감소율 {final_row['평균지급감소율']:.2%}, 평균 지급수익률 {final_row['평균지급수익률']:.2%})",
        "- 차순위 후보: 한 단계만 그 단계 2등으로 바꾼 조합 5개를 Phase 1에서 평가하고, 같은 판단 기준으로 상위 2개를 골랐다.",
        md(dV, ["순위", "바꾼단계", "전략", "평균지급수익률(%)", "비교B0수익률(%)", "조건2_수익률≥B0", "평균지급감소율(%)", "복잡도", "감소율_p95(%)"]),
        f"\n- 차순위1: {spec_name(runner[0])}\n- 차순위2: {spec_name(runner[1])}\n"]

frozen = {
    "frozen_at": "Phase 1 (구성일 2024-01-01, 평가 2024년)",
    "seed": SEED, "n_bundles": N_BUNDLES,
    "decision_rule": "평균 지급감소율 최저 / 평균 지급수익률 ≥ 같은 적격 기준 B0 / 0.5%p 미만 차이는 단순한 쪽 / 동률은 감소율 p95",
    "final": {"name": spec_name(final), "spec": final},
    "runner_up_1": {"name": spec_name(runner[0]), "changed_step": dfV.loc[top2[0], "바꾼단계"], "spec": runner[0]},
    "runner_up_2": {"name": spec_name(runner[1]), "changed_step": dfV.loc[top2[1], "바꾼단계"], "spec": runner[1]},
    "step_choices": {"A": {"1st": A_sel, "2nd": A_2nd}, "B": {"1st": B_sel, "2nd": B_2nd},
                     "C": {"1st": C_sel, "2nd": C_2nd}, "D": {"1st": N_sel, "2nd": N_2nd},
                     "E": {"1st": E_sel, "2nd": E_2nd}},
    "phase1_final_metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in final_row.items() if not isinstance(v, (np.bool_,))},
}
(RESULTS / "frozen_strategy.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log.append("- `frozen_strategy.json` 저장. 이후 Phase 2·2b에서는 이 파일만 읽어 평가하며 변경하지 않는다.")
(LOGS / "02_Phase1_선택.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
