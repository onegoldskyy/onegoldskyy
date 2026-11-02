"""석사논문 수정 보완 분석 전체 재실행 (results/revision/). 약 20분 걸린다.

실행: .venv/bin/python run_revision.py   (윈도우: .venv\\Scripts\\python run_revision.py)
기존 results/·results/extra/·results/extra2/ 파일은 읽기만 한다.
"""
import runpy, sys, time
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
STEPS = ["rev_r0.py", "rev_features.py", "rev_run.py", "rev_combo.py", "rev_age_frontier.py", "rev_figs.py", "rev_second.py"]
t0 = time.time()
for s in STEPS:
    print(f"===== {s} =====", flush=True); t = time.time()
    runpy.run_path(str(SRC / s), run_name="__main__")
    print(f"----- {s} 완료: {time.time() - t:.0f}초", flush=True)
print(f"전체 완료: {time.time() - t0:.0f}초")
