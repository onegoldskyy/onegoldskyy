"""3단계: Phase 2(2025)·2b(2026.1~8) 검증. frozen_strategy.json 만 읽어 평가한다(선택 변경 없음)."""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from config import RESULTS, LOGS, CACHE, PHASES
from engine import run, choose, spec_name, apply_A, PhaseData, summarize
from report_util import save, md

warnings.filterwarnings("ignore")
fz = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
A0 = {"code": "A0", "dec_th": None, "stab_th": None}
B0 = {"code": "B0", "decade": None}
FINAL = fz["final"]["spec"]
STRATS = {
    "최종": FINAL,
    "차순위1": fz["runner_up_1"]["spec"],
    "차순위2": fz["runner_up_2"]["spec"],
    "B0무작위(A0)": {"A": A0, "B": B0, "C": "C0", "N": 20, "E": "E1"},
}
POSTHOC = {"사후탐색:B1[2000년대]": {**FINAL, "B": {"code": "B1", "decade": "2000년대"}}}
PHASE_KEYS = ["P1", "P2", "P2b"]
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}


def b0_yield(pk, spec):
    return run(pk, {"A": spec["A"], "B": B0, "C": "C0", "N": spec["N"], "E": "E1"})["평균지급수익률"]


def single_song(pk, A, price_window=30):
    """단일곡 1개 보유: 적격 곡 각각을 하나의 '묶음'으로 보고 같은 지표 계산."""
    P = PhaseData(pk)
    from engine import bundle_metrics
    ids = apply_A(P.sel, A)
    bundles = np.asarray(ids).reshape(-1, 1)
    m = bundle_metrics(P, bundles, np.ones((len(ids), 1)), price_window)
    return m


def main():
    rows, raw = [], {}
    rank_rows = []
    for pk in PHASE_KEYS:
        res = {}
        for name, s in {**STRATS, **POSTHOC}.items():
            r, (b, w, m) = run(pk, s, return_raw=True)
            r = {"구분": name, **r}
            r["연환산_평균지급수익률"] = r["평균지급수익률"] * ANN[pk]
            res[name] = r
            raw[(pk, name)] = m
        for lab, A in [("단일곡(전체 적격)", A0), ("단일곡(최종 적격기준)", FINAL["A"])]:
            m = single_song(pk, A)
            r = {"구분": lab, "기간": pk, "전략": lab, "적격곡수": len(m), "실행가능": True, **summarize(m)}
            r["묶음수"] = len(m)
            r["연환산_평균지급수익률"] = r["평균지급수익률"] * ANN[pk]
            res[lab] = r
            raw[(pk, lab)] = m
        rows += list(res.values())
        # 판단 기준 순위: 최종·차순위·B0무작위 (사후탐색·단일곡 제외)
        cand = [res[k] for k in STRATS]
        by = {res[k]["전략"]: b0_yield(pk, s) for k, s in STRATS.items()}
        d, _ = choose(cand, by)
        d["감소율순위"] = d["평균지급감소율"].rank(method="min").astype(int)
        rank_rows.append(d.assign(기간=pk))
    with open(CACHE / "phase2_bundle_metrics.pkl", "wb") as f:
        pickle.dump(raw, f)

    tab = pd.DataFrame(rows)
    cols = ["기간", "구분", "전략", "적격곡수", "묶음수", "평균지급수익률", "연환산_평균지급수익률", "지급수익률_중앙", "지급수익률_p5",
            "지급수익률_p95", "지급감소발생비율", "평균지급감소율", "감소율_중앙", "감소율_p95", "곡별감소율가중평균(보조)",
            "월별지급CV_평균", "총수익률_평균", "총수익률_중앙", "총수익률_p5", "총수익률_p95", "손실확률(총수익률<0)",
            "총수익률_집계묶음수", "가격커버리지_평균", "DollarAge_평균", "최대1곡지급비중_평균"]
    tab = tab[cols]
    tab["연환산_평균지급수익률"] = tab["연환산_평균지급수익률"].astype(float)
    d = save(tab.rename(columns={"연환산_평균지급수익률": "평균지급수익률_연환산"}), "3-4_Phase2_검증")
    rk = pd.concat(rank_rows, ignore_index=True)
    rk_s = save(rk[["기간", "구분", "전략", "평균지급수익률", "비교B0수익률", "조건2_수익률≥B0", "평균지급감소율",
                    "감소율_p95", "감소율순위", "순위"]].rename(columns={"순위": "판단기준순위"}), "3-4a_기간별_전략순위")

    # 민감도: (1) 감소율 비교기간을 평가 직전 달력연도로 (2) 가격 기준 7·90일
    sens = []
    for pk in ["P1", "P2"]:
        for name, s in STRATS.items():
            r = run(pk, s, prev_col="eval_prev_alt_pay")
            sens.append({"기간": pk, "구분": name, "민감도": "감소 비교기간=직전 달력연도",
                         "평균지급감소율": r["평균지급감소율"], "지급감소발생비율": r["지급감소발생비율"], "감소율_p95": r["감소율_p95"]})
    for pk in PHASE_KEYS:
        for name, s in STRATS.items():
            for w in (7, 90):
                r = run(pk, s, price_window=w)
                sens.append({"기간": pk, "구분": name, "민감도": f"가격기준 {w}일",
                             "총수익률_평균": r["총수익률_평균"], "총수익률_중앙": r["총수익률_중앙"],
                             "손실확률(총수익률<0)": r["손실확률(총수익률<0)"], "총수익률_집계묶음수": r["총수익률_집계묶음수"],
                             "가격커버리지_평균": r["가격커버리지_평균"]})
    sd = save(pd.DataFrame(sens), "3-4b_검증_민감도")

    # 로그
    log = ["# 3단계. Phase 2·2b 검증\n",
           f"- `frozen_strategy.json`을 읽어 평가했다. 선택 변경 없음. 최종: {fz['final']['name']}, "
           f"차순위1: {fz['runner_up_1']['name']}, 차순위2: {fz['runner_up_2']['name']}.",
           "- B0무작위(A0)는 적격 곡 전체에서 20곡을 동일 금액으로 무작위 구성한 기준선이다.",
           "- '사후탐색:B1[2000년대]'은 Phase 1 B단계에서 0.5%p 규칙으로 탈락한 후보를 참고로 평가한 것이다. "
           "고정 전략이 아니며 순위 판정에서 제외한다.",
           "- 단일곡은 적격 곡 하나씩을 한 묶음으로 본 분포다(묶음 수 = 곡 수).",
           "- Phase 2b 평가기간은 8개월(2026-01~08)이며 감소율은 2025-01~08과 비교한다. 연환산 수익률 = 8개월 수익률 × 12/8 (계절성 때문에 참고값).",
           "- Phase 1은 선택에 쓴 해이므로 표본 내 성과다.\n",
           "## 주요 지표\n"]
    show = ["기간", "구분", "적격곡수", "평균지급수익률(%)", "평균지급수익률_연환산(%)", "지급감소발생비율(%)", "평균지급감소율(%)",
            "감소율_p95(%)", "월별지급CV_평균", "총수익률_평균(%)", "손실확률(총수익률<0)(%)", "총수익률_집계묶음수", "가격커버리지_평균(%)"]
    log.append(d[show].to_markdown(index=False))
    orders = {pk: " > ".join(g.sort_values("판단기준순위")["구분"]) for pk, g in rk_s.groupby("기간", sort=False)}
    dorders = {pk: " > ".join(g.sort_values(["감소율순위", "판단기준순위"])["구분"]) for pk, g in rk_s.groupby("기간", sort=False)}
    log += ["\n## 판단 기준 순위 (최종·차순위·B0무작위)\n",
            "- 이 순위는 네 전략을 한 번에 놓고 6절 기준을 적용한 것이다. N은 복잡도에 포함하지 않으므로, D단계 규칙(20곡 대비 0.5%p)과 달리 "
            "30곡(차순위1)이 1위가 될 수 있다.",
            rk_s.to_markdown(index=False),
            "\n- 판단 기준 순서: " + " / ".join(f"{k}: {v}" for k, v in orders.items()),
            "- 평균 감소율 순서: " + " / ".join(f"{k}: {v}" for k, v in dorders.items()),
            "\n## 민감도\n", sd.to_markdown(index=False)]
    g = tab.set_index(["기간", "구분"])
    f = lambda pk, k, c: g.loc[(pk, k), c] * 100
    log += ["\n## 검증 메모 (자동 생성 수치)\n"]
    for pk in ["P2", "P2b"]:
        log.append(f"- {pk}: 최종 평균 감소율 {f(pk, '최종', '평균지급감소율'):.2f}% vs B0무작위 {f(pk, 'B0무작위(A0)', '평균지급감소율'):.2f}% "
                   f"(차이 {f(pk, '최종', '평균지급감소율') - f(pk, 'B0무작위(A0)', '평균지급감소율'):+.2f}%p), 평균 지급수익률 "
                   f"{f(pk, '최종', '평균지급수익률'):.2f}% vs {f(pk, 'B0무작위(A0)', '평균지급수익률'):.2f}%. "
                   f"단일곡(전체 적격) 평균 감소율 {f(pk, '단일곡(전체 적격)', '평균지급감소율'):.2f}%.")
    log += ["- Phase 2의 감소율 수준이 Phase 1보다 크게 높다. 이는 전략과 무관하게 모든 묶음과 단일곡에서 함께 나타나는 해(年) 공통 충격이다.",
            "- 가격 7일 기준에서는 묶음별 가격 커버리지가 평균 25~47%에 그쳐, 80% 기준을 넘는 묶음이 거의 없다(총수익률 계산 불가). "
            "30일 기준 결과를 주분석으로 쓴다.",
            "- Phase 2b(2026.1~8)에는 모든 전략과 단일곡의 총수익률이 크게 음수다. 이는 가격 하락 때문이며, 지급 지표의 상대 순위와는 별개다.",
            "- 사후탐색 B1[2000년대]은 Phase 2·2b에서도 감소율이 가장 낮았다. 그러나 Phase 1의 사전 규칙으로는 탈락한 후보이고, "
            "검증 결과를 본 뒤 채택하면 미래정보 사용이 된다. 따라서 고정 전략으로 바꾸지 않고 '사후 탐색'으로만 보고한다."]
    (LOGS / "03_Phase2_검증.md").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
