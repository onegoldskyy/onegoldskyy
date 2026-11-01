"""[보조·사후 탐색] 테마 포장 검증 (추가작업 3단계).

모든 묶음: A3 적격 기준(T4만 A1) + 20곡 + 동일 금액. 기존 엔진의 추출·지표 함수를 그대로 쓴다.
판정(결과 보기 전 사전 고정): 세 기간 모두 T0 대비 평균 감소율 악화 ≤ 1%p 이고 지급수익률 하락 ≤ 0.3%p 이면 "작다", 아니면 "크다".
  - 2026.1~8 수익률은 12/8 연환산 값에 기준을 적용한다(원 8개월 값도 보고).
"""
import json
import warnings
import numpy as np
import pandas as pd
from config import RESULTS, PHASES
from engine import (run, apply_A, PhaseData, sample_bundles, weights, bundle_metrics, summarize, _rng)
from rr_events import EXTRA

warnings.filterwarnings("ignore")
TAG = "보조·사후 탐색"
TB = EXTRA / "tables"
fz = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
FINAL = fz["final"]["spec"]
A1 = {"code": "A1", "dec_th": None, "stab_th": None}
PKS = ["P1", "P2", "P2b"]
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}
N = 20
MIN_SUPPLY = 3 * N
KEYS = ["평균지급수익률", "지급감소발생비율", "평균지급감소율", "감소율_p95", "총수익률_평균", "총수익률_p5",
        "손실확률(총수익률<0)", "DollarAge_평균", "최대1곡지급비중_평균"]


def custom(pk, name, pools: list[tuple[pd.Index, int]]):
    """여러 하위 후보군에서 지정 곡 수씩 뽑아 합친 묶음 (C0, E1)."""
    P = PhaseData(pk)
    parts = []
    for j, (pool, n) in enumerate(pools):
        spec = {"A": FINAL["A"], "B": {"code": "B0", "decade": None}, "C": "C0", "N": n, "E": "E1"}
        b, _ = sample_bundles(P.sel, pool, spec, f"theme|{name}|{pk}|{j}")
        parts.append(b)
    bundles = np.concatenate(parts, axis=1)
    w = weights(P.sel, bundles, "E1")
    return summarize(bundle_metrics(P, bundles, w))


def main():
    rows = []
    for pk in PKS:
        P = PhaseData(pk)
        s = P.sel
        el3 = apply_A(s, FINAL["A"])
        el1 = apply_A(s, A1)
        dec = s.loc[el3, "decade"]
        age = s.loc[el3, "age"]
        themes = {
            "T0 기본형(A3 무작위)": ("run", FINAL, [el3]),
            "T1 2000년대": ("run", {**FINAL, "B": {"code": "B1", "decade": "2000년대"}}, [el3[dec == "2000년대"]]),
            "T1 2010년대": ("run", {**FINAL, "B": {"code": "B1", "decade": "2010년대"}}, [el3[dec == "2010년대"]]),
            "T2 15년 이상": ("custom", [(el3[age >= 15], N)], [el3[age >= 15]]),
            "T2 7~15년": ("custom", [(el3[(age >= 7) & (age < 15)], N)], [el3[(age >= 7) & (age < 15)]]),
            "T3 2000년대10+2010년대10": ("custom", [(el3[dec == "2000년대"], 10), (el3[dec == "2010년대"], 10)],
                                      [el3[dec == "2000년대"], el3[dec == "2010년대"]]),
            "T4 A1(7년 이상) 무작위": ("run", {**FINAL, "A": A1}, [el1]),
        }
        # 참고: 공급 부족 연대
        for d_ in ["1990년대 이전", "1990년대", "2020년대"]:
            themes[f"T1 {d_}"] = ("supply_only", None, [el3[dec == d_]])
        for name, (kind, arg, supply_pools) in themes.items():
            sup = [len(p) for p in supply_pools]
            need = [MIN_SUPPLY] if name != "T3 2000년대10+2010년대10" else [30, 30]
            ok = all(a >= b for a, b in zip(sup, need)) and kind != "supply_only"
            r = {"표시": TAG, "기간": pk, "테마": name, "공급곡수": "+".join(map(str, sup)),
                 "비중복20곡묶음수": (min(sup[0] // 10, sup[1] // 10) if len(sup) == 2 else sup[0] // N),
                 "실행가능": ok}
            if ok:
                res = run(pk, arg) if kind == "run" else custom(pk, name, arg)
                r.update({k: res[k] for k in KEYS})
                r["평균지급수익률_연환산"] = res["평균지급수익률"] * ANN[pk]
            rows.append(r)
    D = pd.DataFrame(rows)
    # T0 대비 차이
    t0 = D[D["테마"].str.startswith("T0")].set_index("기간")
    diff_rows = []
    for _, r in D[D["실행가능"] & ~D["테마"].str.startswith("T0")].iterrows():
        b = t0.loc[r["기간"]]
        diff_rows.append({"테마": r["테마"], "기간": r["기간"],
                          **{f"Δ{k}": r[k] - b[k] for k in KEYS + ["평균지급수익률_연환산"]}})
    Dd = pd.DataFrame(diff_rows)
    # 판정
    verdict = []
    for th, g in Dd.groupby("테마", sort=False):
        full = set(g["기간"]) == set(PKS)
        dec_ok = (g["Δ평균지급감소율"] <= 0.01 + 1e-12).all()
        y_ok = (g["Δ평균지급수익률_연환산"] >= -0.003 - 1e-12).all()
        v = "작다" if (full and dec_ok and y_ok) else "크다"
        note = "" if full else "세 기간 중 실행 불가 기간이 있어 '작다' 판정 불가"
        verdict.append({"테마": th, "판정": v, "감소율조건(≤+1%p) 충족": dec_ok, "수익률조건(≥−0.3%p, 연환산) 충족": y_ok,
                        "최대 감소율 악화(%p)": g["Δ평균지급감소율"].max() * 100,
                        "최대 수익률 하락(%p, 연환산)": -g["Δ평균지급수익률_연환산"].min() * 100, "비고": note})
    V = pd.DataFrame(verdict)
    # 세 기간 나란히 (핵심 표)
    wide = Dd.pivot(index="테마", columns="기간", values=["Δ평균지급감소율", "Δ평균지급수익률", "Δ평균지급수익률_연환산", "Δ감소율_p95",
                                                         "Δ총수익률_평균", "Δ손실확률(총수익률<0)", "ΔDollarAge_평균", "Δ최대1곡지급비중_평균"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    order = [t for t in D["테마"].unique() if t in wide.index]
    wide = wide.loc[order]
    pctc = [c for c in wide.columns if "DollarAge" not in c]
    wide[pctc] = wide[pctc] * 100
    wide = wide.round(2).reset_index().merge(V[["테마", "판정"]], on="테마")
    wide.insert(0, "표시", TAG)
    # 저장
    Dp = D.copy()
    for k in [c for c in KEYS + ["평균지급수익률_연환산"] if c != "DollarAge_평균"]:
        Dp[k] = (Dp[k].astype(float) * 100).round(2)
    Dp = Dp.rename(columns={k: k + "(%)" for k in KEYS + ["평균지급수익률_연환산"] if k != "DollarAge_평균"})
    Dp.to_csv(TB / "3-15a_테마포장_수준.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(TB / "3-15_테마포장_비교.csv", index=False, encoding="utf-8-sig")
    V.insert(0, "표시", TAG)
    V.round(3).to_csv(TB / "3-15b_테마포장_판정.csv", index=False, encoding="utf-8-sig")
    Dd.to_pickle(EXTRA / "tables" / "_theme_diff.pkl")
    log = [f"# [{TAG}] 추가작업 3단계: 테마 포장 검증\n",
           "- 모든 묶음: A3 적격 기준 + 20곡 + 동일 금액(T4만 A1 적격 기준). T0·T1·T4는 기존 엔진(run)을 그대로 써서 T0가 기존 최종 전략 수치와 같다.",
           f"- 공급 기준: 테마 후보 {MIN_SUPPLY}곡(3N) 이상, T3는 연대별 30곡(3×10) 이상. 미만이면 공급 수만 보고한다.",
           "- 판정(사전 고정): 세 기간 모두 ΔT0 평균 감소율 ≤ +1%p 그리고 Δ지급수익률 ≥ −0.3%p → '작다'. 2026.1~8은 연환산 수익률에 적용.",
           "- 비중복20곡묶음수 = 공급곡수 ÷ 20(내림). T3는 min(2000년대 공급, 2010년대 공급) ÷ 10.\n",
           "## 수준 (세 기간)\n",
           Dp[["기간", "테마", "공급곡수", "비중복20곡묶음수", "실행가능", "평균지급수익률(%)", "평균지급수익률_연환산(%)", "지급감소발생비율(%)",
               "평균지급감소율(%)", "감소율_p95(%)", "총수익률_평균(%)", "총수익률_p5(%)", "손실확률(총수익률<0)(%)", "DollarAge_평균",
               "최대1곡지급비중_평균(%)"]].to_markdown(index=False),
           "\n## T0 대비 차이 (%p; Dollar Age는 년)\n", wide.drop(columns="표시").to_markdown(index=False),
           "\n## 판정\n", V.drop(columns="표시").round(2).to_markdown(index=False)]
    (EXTRA / "logs" / "13_테마포장.md").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
