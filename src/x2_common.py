"""[보조 분석] 사업계획서 보강(추가 작업 2) 공통 설정.

기존 결과 파일과 frozen_strategy.json은 읽기만 한다. 결과는 results/extra2/ 아래에 저장한다.
장기 보유용 기간 정의는 이 프로세스 안에서만 config.PHASES에 추가한다(config.py 파일은 수정하지 않음).
"""
import json
import pandas as pd
from config import RESULTS, PHASES, months

TAG = "보조 분석"
X2 = RESULTS / "extra2"
for _d in ("tables", "figures", "logs"):
    (X2 / _d).mkdir(parents=True, exist_ok=True)
TB, FG, LG = X2 / "tables", X2 / "figures", X2 / "logs"
FZ = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
FINAL = FZ["final"]["spec"]
A0 = {"code": "A0", "dec_th": None, "stab_th": None}
B0R = {"A": A0, "B": {"code": "B0", "decade": None}, "C": "C0", "N": 20, "E": "E1"}
INDEX_START = "2023-10"           # 1단계: 30일 가격 커버리지 ≥50%인 첫 달(81.1%)
END = pd.Timestamp("2026-08-31")

# 비용 파라미터 (사용자가 확인한 2023년 투자설명서 ANL 수치)
COST = dict(공모총액=40_208_000, 위탁수익금=37_181_848, 신탁보수=2_000_000, 상장수수료=772_912, 기타=253_240,
            상장수수료율=0.02)


def _long_phase(key, form, ltm, prev, lty, base_date):
    if key not in PHASES:
        ev = months(form.strftime("%Y-%m"), "2026-08")
        PHASES[key] = dict(label=f"장기보유 {form.date()}~2026-08-31", form=form, ltm=ltm, prev=prev, lty=lty,
                           eval=ev, eval_prev=ltm, eval_prev_alt=ltm, base_date=base_date, end_date=END)
    return key


def long_phases():
    """구성일 (a) 2023-10-01, (b) 2024-01-01, (c) 2025-01-01 → 종료 2026-08-31.
    선정 변수 규칙은 기존과 같다: LTM = 구성월 2개월 전에 끝나는 12개월, 비교 = 그 앞 12개월, LTY = 구성 전월까지 36개월."""
    a = _long_phase("L2310", pd.Timestamp("2023-10-01"), months("2022-08", "2023-07"), months("2021-08", "2022-07"),
                    months("2020-10", "2023-09"), pd.Timestamp("2023-09-30"))
    b = _long_phase("L2401", pd.Timestamp("2024-01-01"), PHASES["P1"]["ltm"], PHASES["P1"]["prev"], PHASES["P1"]["lty"],
                    PHASES["P1"]["base_date"])
    c = _long_phase("L2501", pd.Timestamp("2025-01-01"), PHASES["P2"]["ltm"], PHASES["P2"]["prev"], PHASES["P2"]["lty"],
                    PHASES["P2"]["base_date"])
    return [a, b, c]


def tsave(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df.copy()
    d.insert(0, "표시", TAG)
    d.to_csv(TB / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df
