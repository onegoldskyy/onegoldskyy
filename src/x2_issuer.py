"""[보조 분석] 3단계: 발행사(뮤직카우) 관점 지표 — 신규 발행, 거래 활동, 발행 여력과 회차 간 겹침."""
import itertools
import warnings
import numpy as np
import pandas as pd
from config import PRICE_CSV, PHASES
from data_prep import load_trades
from engine import apply_A, PhaseData, _rng
from x2_common import TAG, LG, FINAL, tsave

warnings.filterwarnings("ignore")
log = [f"# [{TAG}] 보강 3단계: 발행사 관점 지표\n"]
cohort = PhaseData("P2").sel.index

# ============================================================ 3-1 연도별 신규 발행 (식별 가능 범위 점검)
raw = pd.read_csv(PRICE_CSV, usecols=["date", "provider_song_id", "close_price_krw", "volume", "price_type", "issue_date"],
                  encoding="utf-8-sig")
tr_all = raw[(raw["price_type"] == "traded_daily_close") & (raw["volume"] > 0) & (raw["close_price_krw"] > 0)]
first = pd.to_datetime(tr_all.groupby("provider_song_id")["date"].min())
start = pd.to_datetime(raw["date"].min())
censored = first <= start + pd.Timedelta(days=30)
issue = raw.groupby("provider_song_id")["issue_date"].first()
rows = []
for y in sorted(first.dt.year.unique()):
    m = first.dt.year == y
    rows.append({"연도": y, "첫체결곡수(전체원자료)": int(m.sum()), "그중_수집시작30일내(왼쪽절단)": int((m & censored).sum()),
                 "그중_수집시작30일후": int((m & ~censored).sum()),
                 "issue_date가_그해인곡수": int((pd.to_datetime(issue).dt.year == y).sum())})
for y in [2023, 2024, 2025, 2026]:
    if y not in first.dt.year.unique():
        rows.append({"연도": y, "첫체결곡수(전체원자료)": 0, "그중_수집시작30일내(왼쪽절단)": 0, "그중_수집시작30일후": 0,
                     "issue_date가_그해인곡수": int((pd.to_datetime(issue).dt.year == y).sum())})
T21 = pd.DataFrame(rows).sort_values("연도")
T21["주석"] = "cohort 767곡 안의 수치이며 뮤직카우 전체가 아니다. 공모 정보가 없고 발행일은 전 곡 2023-09-19(현행 증권 일괄 전환)이며, " \
             "첫 체결일은 수집 시작(2021-10-11) 전부터 거래된 곡이 대부분이라 자료상 발행 연도를 식별할 수 없다."
tsave(T21, "3-21_연도별_신규발행_식별점검")
log += ["## 3-1. 연도별 신규 발행 (표 3-21)\n",
        f"- 가격 원자료 시작일 {start.date()}. 첫 체결일이 수집 시작 30일 안이면 수집 전부터 거래된 곡(왼쪽 절단)으로 본다.",
        T21.drop(columns="주석").to_markdown(index=False),
        "\n- **cohort 767곡 안의 수치이며 뮤직카우 전체가 아니다.** 발행일은 전 곡 2023-09-19 하나뿐(현행 증권 일괄 전환)이고, "
        f"첫 체결일 기준으로는 {int(censored.sum())}곡이 수집 시작 30일 안에 이미 거래되어 실제 첫 발행 시점을 알 수 없다. "
        "따라서 이 자료로는 연도별 신규 발행 수를 식별할 수 없다.\n"]

# ============================================================ 3-2 거래 활동
t = load_trades()
t = t[(t["date"] >= "2023-10-01") & (t["date"] <= "2026-08-31")].copy()
t["ym"] = t["date"].dt.to_period("M")
t["value"] = t["volume"] * t["close_price_krw"]
grid = pd.MultiIndex.from_product([cohort, pd.period_range("2023-10", "2026-08", freq="M")], names=["song_id", "ym"])
sm = t.groupby(["song_id", "ym"]).agg(days=("date", "nunique"), value=("value", "sum"), vol=("volume", "sum")).reindex(grid, fill_value=0)
sm = sm.reset_index()


def act(g):
    tot = g.groupby("song_id")["value"].sum().sort_values(ascending=False)
    k = max(1, int(np.ceil(len(tot) * 0.10)))
    return pd.Series({"곡월수": len(g), "거래없는곡월비율(%)": (g["days"] == 0).mean() * 100,
                      "월거래대금_중앙(원)": g["value"].median(), "월거래대금_p90(원)": g["value"].quantile(.9),
                      "월거래대금_평균(원)": g["value"].mean(),
                      "월체결일수_중앙": g["days"].median(), "월체결일수_p90": g["days"].quantile(.9),
                      "상위10%곡_거래대금비중(%)": tot.iloc[:k].sum() / tot.sum() * 100 if tot.sum() > 0 else np.nan,
                      "총거래대금(원)": g["value"].sum()})


monthly = sm.groupby("ym").apply(act).reset_index()
monthly["ym"] = monthly["ym"].astype(str)
sm["year"] = sm["ym"].dt.year
yearly = sm.groupby("year").apply(act).reset_index()
yearly.insert(1, "개월수", sm.groupby("year")["ym"].nunique().values)
whole = act(sm).to_frame().T; whole.insert(0, "year", "2023-10~2026-08"); whole.insert(1, "개월수", sm["ym"].nunique())
yearly = pd.concat([yearly, whole], ignore_index=True)
tsave(monthly.round(2), "3-22a_거래활동_월별")
tsave(yearly.round(2), "3-22_거래활동_연도별")
log += ["## 3-2. 거래 활동 (표 3-22, 3-22a)\n",
        "- 체결 금액 열이 없어 **추정 거래대금 = 일별 체결 수량 × 그날 체결 종가**로 계산했다(하루 전체를 종가로 근사). 월 체결 일수도 함께 보고한다. "
        "사용 조건(현행 증권 이후, 부분 수집일 제외)을 만족한 체결만 쓰며, 체결이 없는 곡-월은 0으로 넣었다. 대상: cohort 767곡, 2023-10~2026-08.",
        "- 상위 10% 곡 비중 = 해당 기간 곡별 거래대금 합계 상위 10% 곡이 전체 거래대금에서 차지하는 비중.",
        yearly.round(2).to_markdown(index=False), ""]

# ============================================================ 3-3 발행 여력과 회차 간 겹침
REPS, MAXUSE, N = 1000, 3, 20
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}
cap, ov = [], []
for pk in ["P1", "P2", "P2b"]:
    P = PhaseData(pk)
    el = np.asarray(apply_A(P.sel, FINAL["A"]))
    n = len(el)
    cap.append({"기간": pk, "구성일": PHASES[pk]["form"].date(), "A3적격곡수": n, "비중복20곡묶음수": n // 20, "비중복30곡묶음수": n // 30,
                "최대3회차중복허용_20곡묶음최대수": (n * MAXUSE) // 20})
    yi = (P.out.loc[el, "eval_pay"] / P.sel.loc[el, "base_price"]).to_numpy() * ANN[pk]
    M = P.mon_eval.loc[el].div(P.sel.loc[el, "base_price"], axis=0).to_numpy()  # 곡 × 월 지급수익률
    for K in (5, 10):
        rng = _rng(f"x2|series|{pk}|{K}")
        Y = np.empty((REPS, K)); overl, mcorr = [], []
        for r in range(REPS):
            use = np.zeros(n, dtype=int); sets = []
            for k in range(K):
                avail = np.flatnonzero(use < MAXUSE)
                pick = rng.choice(avail, N, replace=False)
                use[pick] += 1; sets.append(pick)
                Y[r, k] = yi[pick].mean()
            overl += [len(np.intersect1d(a, b)) for a, b in itertools.combinations(sets, 2)]
            ser = np.array([M[s].mean(0) for s in sets])
            cc = np.corrcoef(ser)
            mcorr.append(cc[np.triu_indices(K, 1)].mean())
        C = np.corrcoef(Y.T)
        ov.append({"기간": pk, "회차수": K, "반복": REPS, "회차간평균겹침곡수": np.mean(overl), "겹침곡수_p95": np.percentile(overl, 95),
                   "한곡최대편입회차": MAXUSE,
                   "회차간연간지급수익률상관_반복간(평균)": C[np.triu_indices(K, 1)].mean(),
                   "회차간월별지급상관_같은반복내(평균)": np.mean(mcorr),
                   "회차연간수익률_평균(%)": Y.mean() * 100, "회차연간수익률_회차간표준편차_평균(%p)": Y.std(axis=1, ddof=1).mean() * 100})
T23a, T23b = pd.DataFrame(cap), pd.DataFrame(ov).round(3)
NOTE = "동일 금액 구성에서는 곡당 투자 금액(지분 크기)이 성과 지표에 영향을 주지 않는다. 공급 자금이 부족하면 곡 수가 아니라 곡당 지분을 줄이면 된다."
T23 = T23b.merge(T23a, on="기간")
T23["주석"] = NOTE
tsave(T23, "3-23_발행여력_회차겹침")
log += ["## 3-3. 발행 여력과 회차 간 겹침 (표 3-23)\n",
        "- 같은 구성일에 5개·10개 회차를 무작위로 발행(회차 안은 서로 다른 20곡, 한 곡은 최대 3개 회차까지), 1,000번 반복.",
        "- '반복 간 상관' = 1,000번 반복에서 회차 i와 j의 연간 지급수익률(2026은 연환산)의 상관(지시 정의). '같은 반복 내 월별 상관' = 한 번 발행한 회차들의 월별 지급수익률 시계열 상관(보조).",
        T23a.to_markdown(index=False), "", T23b.to_markdown(index=False),
        "\n- 해석: 회차 간 평균 겹침(약 0.9곡)은 무작위 추출의 기대값(20×20÷적격 곡 수)과 같다. 반복 간 연간 수익률 상관이 0에 가까운 것은 "
        "겹침이 작아 한 회차의 곡 구성이 다른 회차 성과를 거의 좌우하지 않는다는 뜻이다. 반면 한 번에 발행한 회차들의 월별 지급은 같은 해 공통 충격 때문에 "
        "상관이 0.5~0.8로 높다. 즉 회차를 늘려도 시장 공통 위험은 회차 사이에서 분산되지 않는다.",
        f"\n- {NOTE}\n"]
(LG / "22_발행사지표.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
