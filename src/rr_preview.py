"""[보조·사후 탐색] 역주행 정의 적용 미리보기 (추가작업 1단계)."""
import pandas as pd
from rr_events import detect, eligible_song_months, EXTRA

ev = detect()
el = eligible_song_months()
out = ev.copy()
out.insert(0, "표시", "보조·사후 탐색")
out.round(3).to_csv(EXTRA / "tables" / "3-11a_역주행_사건목록.csv", index=False, encoding="utf-8-sig")
top = ev.sort_values("mult_peak_6m", ascending=False).head(10)
sens = []
for m, c, a in [(3, 2, 3), (2, 2, 3), (5, 2, 3), (3, 1, 3), (3, 3, 3), (3, 2, 2), (3, 2, 5)]:
    e = detect(m, c, a)
    sens.append({"배수": m, "연속개월": c, "최소연령(년)": a, "사건수": len(e), "사건곡수": e["song_id"].nunique() if len(e) else 0,
                 "기본정의": (m, c, a) == (3, 2, 3)})
sens = pd.DataFrame(sens)
log = ["# [보조·사후 탐색] 추가작업 1단계: 역주행 정의 미리보기\n",
       "- 전송 지급액 = amount_per_share × pct_transmission/100. `pct_digital_audio_transmission`은 cohort 767곡 52,119행 전체에서 0이므로 합산 여부와 무관하다(합산하지 않음).",
       "- 기본 정의: 발매 3년 이상, 전송 지급액 ≥ 직전 12개월 평균×3이 2개월 연속. 연속 판정 기준선은 첫 달 직전 12개월 평균으로 고정. "
       "기준선이 0이거나 전체 곡-월 기준선(>0) 분포의 하위 10% 이하이면 제외. 같은 곡의 12개월 내 재발은 하나로 묶음.",
       f"- 하위 10% 기준값: 1주당 월 {ev.attrs['low_cut']:.2f}원. 기준선 12개월이 필요하므로 사건 시작월은 2022-01~2026-07 범위다.",
       f"- 판정 가능 곡(한 달이라도 조건 충족) {int(el.any(axis=1).sum())}곡, 판정 가능 곡-월 {int(el.to_numpy().sum()):,}개.",
       f"- **사건 {len(ev)}건, {ev['song_id'].nunique()}곡.** 연도별: {ev['event_year'].value_counts().sort_index().to_dict()}",
       "\n## 대표 사례 10곡 (사건 후 6개월 내 최고 배수 순)\n",
       top[["title", "singer", "release_year", "event_ym", "age_at_event", "baseline_12m", "mult_first", "mult_min_consec", "mult_peak_6m"]]
       .rename(columns={"title": "곡명", "singer": "가수", "release_year": "발매연도", "event_ym": "사건시작월", "age_at_event": "사건시곡연령",
                        "baseline_12m": "기준선(원/주/월)", "mult_first": "첫달배수", "mult_min_consec": "연속구간최소배수", "mult_peak_6m": "6개월내최고배수"})
       .round(2).to_markdown(index=False),
       "\n## 정의 민감도\n", sens.to_markdown(index=False),
       "\n- 연속 1개월 정의는 사건이 401건으로 급증한다. 한 달짜리 정산 몰림이 많이 섞이므로 기본 정의(2개월 연속)를 유지한다."]
(EXTRA / "logs" / "11_역주행_미리보기.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
