"""분석 전체에서 공유하는 경로, 기간, 상수 정의."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "original"
ROYALTY_CSV = RAW / "02_royalty" / "musicow_monthly_raw.csv"
PRICE_CSV = RAW / "01_price" / "prices_daily_767.csv"
COHORT_CSV = RAW / "03_metadata" / "cohort_767.csv"
MATCH_CSV = RAW / "03_metadata" / "matching_767.csv"
RECOVERY_CSV = DATA / "raw" / "recovery" / "year_end_actual_trades_662.csv"
DOCX = DATA / "음악수익증권 논문 구상_이빈아.docx"

CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
for _p in (CACHE, RESULTS, TABLES, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261101
N_BUNDLES = 3000
ISSUE_DATE = pd.Timestamp("2023-09-19")  # 현행 증권 발행일 (이전 지급은 소급자료)

PCT_COLS = ["pct_broadcast", "pct_transmission", "pct_reproduction", "pct_performance",
            "pct_overseas", "pct_other", "pct_digital_audio_transmission"]
SOURCE_LABEL = {"pct_broadcast": "방송", "pct_transmission": "전송", "pct_reproduction": "복제",
                "pct_performance": "공연", "pct_overseas": "해외", "pct_other": "기타",
                "pct_digital_audio_transmission": "디지털음성송신"}


def months(start: str, end: str) -> list[str]:
    """'YYYY-MM' 문자열 목록 (양 끝 포함)."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


# 3절 기간표.
# eval_prev: 평가기간 지급감소율의 비교기간. 기본안 = 선정용 LTM (원고 표 5와 동일 정의, 재현 확인됨).
#            2b는 지시대로 전년 같은 달(2025-01~08).
# eval_prev_alt: 민감도용 = 평가기간 바로 앞 달력 12개월.
PHASES = {
    "P1": dict(label="Phase 1 (선택, 2024)", form=pd.Timestamp("2024-01-01"),
               ltm=months("2022-12", "2023-11"), prev=months("2021-12", "2022-11"),
               lty=months("2021-01", "2023-12"),
               eval=months("2024-01", "2024-12"), eval_prev=months("2022-12", "2023-11"),
               eval_prev_alt=months("2023-01", "2023-12"),
               base_date=pd.Timestamp("2023-12-31"), end_date=pd.Timestamp("2024-12-31")),
    "P2": dict(label="Phase 2 (검증, 2025)", form=pd.Timestamp("2025-01-01"),
               ltm=months("2023-12", "2024-11"), prev=months("2022-12", "2023-11"),
               lty=months("2022-01", "2024-12"),
               eval=months("2025-01", "2025-12"), eval_prev=months("2023-12", "2024-11"),
               eval_prev_alt=months("2024-01", "2024-12"),
               base_date=pd.Timestamp("2024-12-31"), end_date=pd.Timestamp("2025-12-31")),
    "P2b": dict(label="Phase 2b (추가 검증, 2026.1~8)", form=pd.Timestamp("2026-01-01"),
                ltm=months("2024-12", "2025-11"), prev=months("2023-12", "2024-11"),
                lty=months("2023-01", "2025-12"),
                eval=months("2026-01", "2026-08"), eval_prev=months("2025-01", "2025-08"),
                eval_prev_alt=months("2025-01", "2025-08"),
                base_date=pd.Timestamp("2025-12-31"), end_date=pd.Timestamp("2026-08-31")),
}
