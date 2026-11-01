"""1단계: 자료 점검 (기간별 표본 수, 표 5 재현, 가격 확보율, ym 판단)."""
import numpy as np
import pandas as pd
from config import PHASES, TABLES, LOGS, CACHE, RECOVERY_CSV
from data_prep import load_royalty, load_cohort, load_trades, last_trade
from features import build_phase

TABLE5_MANUSCRIPT = pd.DataFrame({
    "연령구간": ["<7년", "7~15년", "15년+"], "원고_곡수": [164, 315, 183],
    "원고_평균지급수익률(%)": [4.87, 6.38, 6.89], "원고_지급감소곡비중(%)": [82.32, 78.41, 78.14],
    "원고_평균감소율(%p)": [21.42, 16.15, 16.31]})


def main():
    log = ["# 1단계. 자료 점검\n"]
    r_all = load_royalty(cohort_only=False)
    coh = load_cohort()
    t = load_trades()
    meta = (CACHE / "trades_meta.txt").read_text().strip().replace("\n", ", ")
    log.append(f"- 지급자료: {len(r_all):,}행, {r_all.song_id.nunique()}곡, 기간 {r_all.ym.min()}~{r_all.ym.max()}, "
               f"(song_id, ym) 중복 {r_all.duplicated(['song_id', 'ym']).sum()}건, 권리유형 {r_all.rights_type.unique().tolist()}.")
    log.append(f"- cohort: {len(coh)}곡, 발행일 {coh.issue_date.unique().tolist()}, 지급이력 내부 결측 월 합계 "
               f"{int(coh.missing_months_inside_range.sum())}. 지급 시작월이 2021-01이 아닌 곡 {int((coh.earliest_month != '2021-01').sum())}곡.")
    log.append(f"- 가격자료: {meta} (사용 조건: price_type=traded_daily_close, volume>0, close_price_krw>0, "
               f"pre_current_security_issue=0, is_partial_collection_day=0). 필터 후 {t.song_id.nunique()}곡, "
               f"{t.date.min().date()}~{t.date.max().date()}. 2026-08-31 이하 {int((t.date <= '2026-08-31').sum()):,}행 "
               f"(recovery_verification.txt의 116,556행과 비교).")
    log.append("- 기준일 체결가 규칙: 기준일 당일 포함 이전 마지막 체결, 경과일 0~N 달력일 (recovery_verification.txt와 동일).")

    # 기간별 표본 흐름
    rows, cov = [], []
    sels = {}
    for pk, P in PHASES.items():
        sel, out = build_phase(pk)
        sels[pk] = (sel, out)
        n0 = len(sel)
        a = sel["ltm_complete"]; b = a & sel["prev_complete"]; c = b & sel["base_price"].notna()
        rows.append({"구분": P["label"], "구성일": P["form"].date(), "cohort": n0,
                     "선정용LTM완전": int(a.sum()), "+비교12개월완전": int(b.sum()),
                     "+기초가격30일내(적격)": int(c.sum()),
                     "LTY가용개월(적격, 최소~최대)": f"{int(sel.loc[c, 'lty_months'].min())}~{int(sel.loc[c, 'lty_months'].max())}",
                     "평가기간지급완전(적격중)": int((c & out["eval_months_complete"]).sum()),
                     "소급자료사용월수": len(sel.attrs["retro_months"]),
                     "소급자료월범위": (f"{sel.attrs['retro_months'][0]}~{sel.attrs['retro_months'][-1]}"
                                     if sel.attrs["retro_months"] else "-")})
        for w in (7, 30, 90):
            nb = last_trade(t, P["base_date"], w).index.isin(sel.index).sum()
            ne_all = last_trade(t, P["end_date"], w).index.isin(sel.index).sum()
            ne_el = out.loc[c, f"end_price_{w}"].notna().sum()
            cov.append({"구분": P["label"], "기준(일)": w,
                        "기초가격확보(767중)": int(nb), "기초확보율(%)": round(nb / n0 * 100, 2),
                        "기말가격확보(767중)": int(ne_all), "기말확보율(%)": round(ne_all / n0 * 100, 2),
                        "기말가격확보(적격중)": int(ne_el), "기말확보율_적격(%)": round(ne_el / c.sum() * 100, 2)})
    flow = pd.DataFrame(rows)
    cov = pd.DataFrame(cov)
    flow.to_csv(TABLES / "3-1_기간별_표본구성.csv", index=False, encoding="utf-8-sig")
    cov.to_csv(TABLES / "3-1b_기간별_가격확보율.csv", index=False, encoding="utf-8-sig")
    log.append("\n## 기간별 표본\n")
    log.append(flow.to_markdown(index=False))
    log.append("\n## 가격 확보율\n")
    log.append(cov.to_markdown(index=False))

    # 662곡 일치
    sel2, out2 = sels["P2"]
    el2 = sel2.index[sel2["eligible"]]
    rec = pd.read_csv(RECOVERY_CSV, encoding="utf-8-sig")
    same = set(el2) == set(rec["곡ID"])
    log.append(f"\n- Phase 2 적격 {len(el2)}곡과 year_end_actual_trades_662.csv의 662곡 ID 일치 여부: **{same}**.")

    # 표 5 재현
    s = sel2.loc[el2]; o = out2.loc[el2]
    y = o["eval_pay"] / s["base_price"]
    dec = (1 - o["eval_pay"] / o["eval_prev_pay"]).clip(lower=0)
    dec_alt = (1 - o["eval_pay"] / o["eval_prev_alt_pay"]).clip(lower=0)
    df = pd.DataFrame({"연령구간": s["age_bin"].astype(str), "y": y, "d": dec, "dd": o["eval_pay"] < o["eval_prev_pay"],
                       "d_alt": dec_alt, "dd_alt": o["eval_pay"] < o["eval_prev_alt_pay"]})
    g = df.groupby("연령구간").agg(재현_곡수=("y", "size"), y=("y", "mean"), dd=("dd", "mean"), d=("d", "mean"),
                                  dd_alt=("dd_alt", "mean"), d_alt=("d_alt", "mean")).reset_index()
    g["재현_평균지급수익률(%)"] = (g.pop("y") * 100).round(2)
    g["재현_지급감소곡비중(%)"] = (g.pop("dd") * 100).round(2)
    g["재현_평균감소율(%p)"] = (g.pop("d") * 100).round(2)
    g["민감도_달력연도비교_감소곡비중(%)"] = (g.pop("dd_alt") * 100).round(2)
    g["민감도_달력연도비교_평균감소율(%p)"] = (g.pop("d_alt") * 100).round(2)
    t5 = TABLE5_MANUSCRIPT.merge(g, on="연령구간")
    for a, b in [("원고_곡수", "재현_곡수"), ("원고_평균지급수익률(%)", "재현_평균지급수익률(%)"),
                 ("원고_지급감소곡비중(%)", "재현_지급감소곡비중(%)"), ("원고_평균감소율(%p)", "재현_평균감소율(%p)")]:
        t5["일치_" + a.split("_")[1]] = np.isclose(t5[a], t5[b], atol=0.005)
    t5.to_csv(TABLES / "3-2_원고표5_재현.csv", index=False, encoding="utf-8-sig")
    log.append("\n## 원고 표 5 재현 (2025년, 662곡)\n")
    log.append(t5.to_markdown(index=False))
    log.append("\n- 정의: 곡 연령=(2025-01-01−release_date)/365.25, 지급수익률=2025년 지급 합÷2024-12-31 이전 30일 내 체결가, "
               "지급감소율=max(0, 1−2025년 지급÷선정용 LTM(2023-12~2024-11)).")
    log.append("- **해석 결정(기본안)**: 6절 '직전 12개월 지급'을 선정용 LTM으로 둔다. 이 정의에서만 원고 표 5의 감소 지표가 "
               "정확히 재현되었다. 대안(평가기간 바로 앞 달력연도 12개월, 예: 2024-01~12)은 감소 지표를 1~4%p 높게 만들며 "
               "민감도로 보고한다. Phase 2b는 지시대로 전년 같은 달(2025-01~08)과 비교한다.")

    # ym 판단
    rc = load_royalty()
    mb = rc.groupby("ym").apply(lambda d: np.average(d.pct_broadcast, weights=d.amount_per_share.clip(lower=1e-9)))
    mon = pd.Series(mb.values, index=pd.PeriodIndex(mb.index, freq="M").month)
    bym = mon.groupby(level=0).mean().round(1)
    zero_months = int((mb < 0.5).sum())
    pd.DataFrame({"월": bym.index, "방송비중_평균(%)": bym.values}).to_csv(
        TABLES / "3-1c_월별_방송비중_ym판단.csv", index=False, encoding="utf-8-sig")
    log.append("\n## ym의 의미 판단\n")
    log.append(f"- 지급액 가중 방송 비중이 0.5% 미만인 달이 전체 {len(mb)}개월 중 {zero_months}개월이고, 특정 달(예: 2·11·12월)에 "
               f"몰린다. 월별 평균(달별): {bym.to_dict()}.")
    log.append("- 방송 사용 자체는 매달 발생하므로, 이 계절 패턴은 저작권 단체의 분배(정산) 일정을 반영한 것으로 보인다. "
               "따라서 `ym`은 **지급(정산)월**일 가능성이 높다고 판단한다. 단, 자료에 귀속월 정보가 없어 확정할 수 없다. "
               "지급월이라면 구성일 직전 달의 지급은 구성 시점에 알 수 있는 정보이므로 미래정보 문제는 없다.")
    log.append("- LTY 기간은 지시대로 LTM보다 한 달 늦게 끝난다(예: Phase 1 LTM 2022-12~2023-11, LTY 2021-01~2023-12). "
               "두 기간 모두 구성일 이전이다.")
    log.append("- 2023-09-19 이전 지급(소급자료)은 선정 정보로만 쓴다(위 표 '소급자료사용월수').")
    (LOGS / "01_자료점검.md").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
