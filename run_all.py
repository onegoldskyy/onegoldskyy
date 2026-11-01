"""전체 재실행: 자료 점검 → Phase 1 선택 → Phase 2·2b 검증 → 추가 분석 → 그림 → analysis_log.md.

실행 방법 (프로젝트 폴더에서):
    맥/리눅스:  .venv/bin/python run_all.py
    윈도우:     .venv\\Scripts\\python run_all.py
"""
import platform
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

STEPS = [("1. 자료 점검", "step1_check.py"),
         ("2. Phase 1 선택", "phase1_select.py"),
         ("3. Phase 2·2b 검증", "phase2_validate.py"),
         ("4. 추가 분석", "extra_analysis.py"),
         ("5. 그림", "figures.py")]

EXTRA_STEPS = [("6-1. [보조] 역주행 정의 미리보기", "rr_preview.py"),
               ("6-2. [보조] 역주행 분석", "rr_analysis.py"),
               ("6-3. [보조] 테마 포장 검증", "theme_check.py"),
               ("6-4. [보조] 추가 그림", "rr_figures.py")]

HEADER = """# 분석 로그 (analysis_log.md)

이 파일은 `run_all.py`가 자동으로 만든다. 단계별 원본 로그는 `results/logs/`에 있다.

## 실행 환경
- 실행 시각: {when}
- Python {py}, pandas {pd}, numpy {np}, matplotlib {mpl}, statsmodels {sm}, scipy {sp}
- seed = 20261101, 단계별 가상 묶음 3,000개
- 원본 `data/`는 읽기만 한다. 가격 원자료는 필요한 열만 읽어 `cache/trades_filtered.pkl`로 캐시한다.

## 원칙 준수
- 선정 변수는 구성일 이전 정보만 사용한다(`features.py`의 `sel`). 평가기간 지급·기말가격은 별도 표(`out`)에 두고 성과 계산에서만 쓴다.
- 모든 선택은 Phase 1(2024)에서만 했고 `results/frozen_strategy.json`에 저장했다. Phase 2·2b 스크립트는 이 파일을 읽기만 한다. 선택 변경 없음.
- Phase 2·2b에서 추가로 본 B1[2000년대]은 '사후 탐색'으로 표시하고 순위 판정에서 제외했다.
- 가상 묶음 3,000개는 같은 해 환경을 공유하는 비독립 표본이므로 유의성 검정을 하지 않았다.

## 해석이 갈린 지점: 기본안과 대안
| 항목 | 기본안(사용) | 대안(기록) | 근거 |
|---|---|---|---|
| 평가기간 지급감소율의 비교기간 | 선정용 LTM (예: 2025 평가 ↔ 2023-12~2024-11) | 평가 직전 달력 12개월 (3-4b 민감도) | 원고 표 5가 이 정의에서만 전부 재현됨 |
| Phase 2b 비교기간 | 전년 같은 달 2025-01~08 | – | 지시문 |
| `ym`의 의미 | 지급(정산)월로 판단 | 귀속월 | 방송 비중이 특정 달에 몰림(3-1c). 확정 불가 |
| 묶음 지급감소율 | 묶음 합계 지급 기준(투자금 가중 보유주식 × 지급) | 곡별 감소율의 투자금 가중 평균(보조 열) | 투자자가 실제 받는 현금흐름 |
| A단계 조건② 비교 기준 | A0 무작위 | – | A 후보는 모두 B0라 '같은 적격 기준 B0'가 자기 자신 |
| A2·A3 기준값 | A2 {{0,5,10%}}, A3 {{0.1,0.2}} 각각 별도 후보, A1+A2도 세 기준값 각각 | – | 지시문 |
| B1 대표 연대 | 공급 ≥3N 연대 중 판단 기준 1위 | – | 지시문 |
| B2·B3 | 상위 3N곡 중 N곡 무작위 | – | 지시문 |
| B4 층 배분 | 주요 수익원 구성비 × N, 최대잉여법 | – | – |
| C1 상한 'N/10' | 올림(5·10곡→1, 20곡→2, 30곡→3) | 내림(5곡→0) | 5곡에서 방송 중심 곡 전면 배제를 피함 |
| C 후보 | C0, C1, C2, C1+C2 | – | 결합도 비교 |
| D 규칙 | 20곡과 평균 감소율 차이 0.5%p 미만인 최소 곡 수 | 4개 곡 수 중 최저와 0.5%p 이내인 최소 곡 수 (결과 동일: 20곡) | 계획 단계에서 확인 |
| E3 '지급 규모' | 곡 전체 LTM(1주당 지급 × 월별 주식 수) | 1주당 LTM 비례 | 1주당 금액은 주식 분할에 따라 달라지는 임의 단위 |
| E4 | Ledoit-Wolf 축소 공분산(최근 36개월 월별 지급수익률), 곡당 1~15%, N<7이면 상한 1/N | 표본 공분산 | 36개월로 20×20 공분산을 안정적으로 추정하기 어려움 |
| 복잡도(단순성) | 규칙 수 합계: A2·A3·A1 각 1, A1+A2 2, B0 0·그 외 1, C1·C2 1·결합 2, E1 0·E2·E3 1·E4 2 | – | '적은 규칙, 동일 금액 우선' |
| 동률(④) 적용 | 0.5%p 이내 후보 중 복잡도가 같으면 감소율 p95 낮은 쪽 | 평균 감소율 낮은 쪽 | 지시문 ④ |
| 차순위 2개 | 한 단계만 그 단계 2등으로 바꾼 조합 5개 중 판단 기준 상위 2개 | – | – |
| 총수익률 | 기말가격 30일 내 없는 곡 제외 후 재가중, 커버리지 80% 미만 묶음 제외 | 7·90일(3-4b) | 지시문 |
| 바스켓 30일 확률 | 가격조건 없는 후보군(지급이력·안정성만)에서 계산 | – | 적격 후보군은 정의상 30일 체결 100% |
| 안정성 프리미엄 | log(가격/LTM) 외에 log(가격/LTY) 모형 추가 | 배수 수준(1·99% 절단) | 분모가 LTM이면 LTM 일시 감소가 배수를 기계적으로 키움 |

## 표본 수 변화
`results/tables/3-1_기간별_표본구성.csv` 참조 (cohort 767 → 지급이력 완전 767 → 기초가격 30일 내: Phase 1 657, Phase 2 662, Phase 2b 594). 최종 전략 적격(A3): 437 / 456 / 394곡.

## 사후 점검·사후 탐색 기록
- Phase 1 선택 후 seed 10개 잡음 점검(02 로그 끝). 선택은 바꾸지 않았다.
- B1[2000년대]를 Phase 2·2b에서 참고 평가(03 로그). 고정 전략 아님.
- 안정성 프리미엄 M4(가격/LTY)는 4단계 검토 중 추가한 모형이다(선정과 무관).
"""


def main():
    t0 = time.time()
    for label, script in STEPS:
        print(f"\n===== {label} ({script}) =====", flush=True)
        s = time.time()
        runpy.run_path(str(SRC / script), run_name="__main__")
        print(f"----- {label} 완료: {time.time() - s:.1f}초", flush=True)
    import pandas, numpy, matplotlib, statsmodels, scipy
    head = HEADER.format(when=time.strftime("%Y-%m-%d %H:%M:%S"), py=platform.python_version(),
                         pd=pandas.__version__, np=numpy.__version__, mpl=matplotlib.__version__,
                         sm=statsmodels.__version__, sp=scipy.__version__)
    logs = sorted((ROOT / "results" / "logs").glob("*.md"))
    body = "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in logs)
    new = head + "\n---\n\n" + body
    path = ROOT / "results" / "analysis_log.md"
    # 실행 시각 한 줄만 다르면 기존 파일을 그대로 둔다(결과 불변이면 파일도 불변).
    strip = lambda txt: "\n".join(l for l in txt.splitlines() if not l.startswith("- 실행 시각:"))
    if not path.exists() or strip(path.read_text(encoding="utf-8")) != strip(new):
        path.write_text(new, encoding="utf-8")
        print("results/analysis_log.md 갱신(내용 변경)")
    else:
        print("results/analysis_log.md 내용 동일 → 파일 유지")

    # ---------------- 6. 보조·사후 탐색 (기존 결과·frozen_strategy.json은 읽기만 함)
    for label, script in EXTRA_STEPS:
        print(f"\n===== {label} ({script}) =====", flush=True)
        s = time.time()
        runpy.run_path(str(SRC / script), run_name="__main__")
        print(f"----- {label} 완료: {time.time() - s:.1f}초", flush=True)
    xl = sorted((ROOT / "results" / "extra" / "logs").glob("*.md"))
    (ROOT / "results" / "extra" / "analysis_log_extra.md").write_text(
        "# [보조·사후 탐색] 추가 분석 로그\n\n이 파일은 `run_all.py`가 자동으로 만든다. 기존 분석 결과와 `frozen_strategy.json`은 수정하지 않았다.\n\n---\n\n"
        + "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in xl), encoding="utf-8")
    print(f"\n전체 완료: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
