"""[보조 분석] 2단계: 총수익률 분해, 시장 가격지수, 장기 보유 성과, 회복 기간, 현금 기준 누적 지급."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figures import BLUE, AQUA, INK, GRAY  # 한글 폰트·스타일 설정(그림 생성 전에 불러와야 함)
from config import PHASES
from data_prep import load_trades, last_trade
from engine import run, apply_A, PhaseData, bundle_metrics, summarize, COVERAGE_MIN
from x2_common import TAG, TB, FG, LG, FINAL, B0R, INDEX_START, END, long_phases, tsave

warnings.filterwarnings("ignore")
PKS = ["P1", "P2", "P2b"]
ANN = {"P1": 1.0, "P2": 1.0, "P2b": 12 / 8}
log = [f"# [{TAG}] 보강 2단계: 총수익률 분해와 보유기간 성과\n",
       "- 1단계 점검: 가격 자료에 체결 수량(volume)만 있고 금액·건수는 없음. 공모 정보 없음, 발행일은 767곡 모두 2023-09-19(일괄 전환). "
       f"30일 가격 커버리지 ≥50% 첫 달 = {INDEX_START}(월말 81.1%, 2023-09말 39.4%).",
       "- 가격 규칙·커버리지 규칙(기말가격 없는 곡 제외 후 재가중, 커버리지 80% 미만 묶음 제외)은 기존과 같다.\n"]


def decompose(P, bundles, w):
    """총수익률 = 지급수익 + 가격수익 (기말가격 확보 곡으로 재가중, 커버리지<80% 묶음 제외)."""
    nb, N = bundles.shape
    flat = bundles.ravel()
    g = lambda s: s.reindex(flat).to_numpy(dtype=float).reshape(nb, N)
    b0, p1, ev = g(P.sel["base_price"]), g(P.out["end_price_30"]), g(P.out["eval_pay"])
    have = np.isfinite(p1) & np.isfinite(b0)
    cov = (w * have).sum(1)
    pay = (w * np.where(have, ev / b0, 0)).sum(1) / np.where(cov > 0, cov, np.nan)
    prc = (w * np.where(have, (np.nan_to_num(p1) - b0) / b0, 0)).sum(1) / np.where(cov > 0, cov, np.nan)
    keep = cov >= COVERAGE_MIN
    return pd.DataFrame({"pay": np.where(keep, pay, np.nan), "price": np.where(keep, prc, np.nan),
                         "tr": np.where(keep, pay + prc, np.nan), "cov": cov})


def stats(x, pre):
    x = pd.Series(x).dropna()
    return {f"{pre}_평균": x.mean(), f"{pre}_중앙": x.median(), f"{pre}_p5": x.quantile(.05)}


# ============================================================ 2-1 총수익률 분해
rows, keep_bundles = [], {}
for pk in PKS:
    P = PhaseData(pk)
    for lab, spec in [("최종 묶음", FINAL), ("B0 무작위 20곡", B0R)]:
        r, (b, w, m) = run(pk, spec, return_raw=True)
        D = decompose(P, b, w)
        assert np.allclose(D["tr"].dropna().to_numpy(), m["total_return"].dropna().to_numpy()), "총수익률 불일치"
        keep_bundles[(pk, lab)] = (b, w, D)
        rows.append({"기간": pk, "구분": lab, "집계수": int(D["tr"].notna().sum()),
                     **stats(D["tr"], "총수익률"), **stats(D["pay"], "지급수익"), **stats(D["price"], "가격수익"),
                     "손실확률": (D["tr"].dropna() < 0).mean(), "가격수익<0 비율": (D["price"].dropna() < 0).mean()})
    ids = np.asarray(apply_A(P.sel, FINAL["A"])).reshape(-1, 1)
    D = decompose(P, ids, np.ones_like(ids, dtype=float))
    rows.append({"기간": pk, "구분": "단일곡(최종 적격기준)", "집계수": int(D["tr"].notna().sum()),
                 **stats(D["tr"], "총수익률"), **stats(D["pay"], "지급수익"), **stats(D["price"], "가격수익"),
                 "손실확률": (D["tr"].dropna() < 0).mean(), "가격수익<0 비율": (D["price"].dropna() < 0).mean()})
T16 = pd.DataFrame(rows)
pc = [c for c in T16.columns if c not in ("기간", "구분", "집계수")]
T16p = T16.copy(); T16p[pc] = (T16p[pc] * 100).round(2); T16p = T16p.rename(columns={c: c + "(%)" for c in pc})
tsave(T16p, "3-16_총수익률분해")
log += ["## 2-1. 총수익률 분해 (표 3-16)\n",
        "- 같은 곡(기말가격 확보 곡)으로 지급수익과 가격수익을 계산하므로 두 값의 합은 기존 표 3-4의 총수익률과 정확히 같다(코드에서 일치 확인).",
        "- 평균과 p5는 성분별로 따로 계산한 값이어서, 지급수익 p5 + 가격수익 p5 ≠ 총수익률 p5이다.",
        T16p.to_markdown(index=False), ""]

# ============================================================ 2-2 시장 가격지수
t = load_trades()
cohort = PhaseData("P2").sel.index
mends = pd.date_range(f"{INDEX_START}-01", END, freq="ME")
PM = pd.DataFrame({d: last_trade(t, d, 30)["price"].reindex(cohort) for d in mends})  # 곡 × 월말 가격
idx_rows, lvl = [], 100.0
for i, d in enumerate(mends):
    cov_now = PM[d].notna().mean()
    if i == 0:
        idx_rows.append({"월말": d.date(), "지수": lvl, "월수익률": np.nan, "가격보유곡비율": cov_now, "수익률계산곡수": np.nan,
                         "수익률계산곡비율": np.nan}); continue
    both = PM[d].notna() & PM[mends[i - 1]].notna()
    r = (PM.loc[both, d] / PM.loc[both, mends[i - 1]] - 1).mean()
    lvl *= 1 + r
    idx_rows.append({"월말": d.date(), "지수": lvl, "월수익률": r, "가격보유곡비율": cov_now, "수익률계산곡수": int(both.sum()),
                     "수익률계산곡비율": both.mean()})
IDX = pd.DataFrame(idx_rows).set_index("월말")
mret = IDX["월수익률"]
idxlvl = IDX["지수"]
cmp, r2rows = [], []
for pk in PKS:
    b0d, e1d = PHASES[pk]["base_date"].date(), PHASES[pk]["end_date"].date()
    mkt = idxlvl[e1d] / idxlvl[b0d] - 1
    D = keep_bundles[(pk, "최종 묶음")][2]
    bp = D["price"].mean()
    # 묶음별 월 가격수익률(동일가중, 연속 두 달 가격 보유 곡) → 시장 월수익률에 회귀한 R²
    b, w, _ = keep_bundles[(pk, "최종 묶음")]
    ms = [d for d in mends if b0d < d.date() <= e1d]
    prev = [mends[list(mends).index(d) - 1] for d in ms]
    X = mret.loc[[d.date() for d in ms]].to_numpy()
    r2 = []
    for bi in range(b.shape[0]):
        sub = PM.loc[b[bi]]
        y = np.array([np.nanmean((sub[d] / sub[p] - 1).to_numpy()) for d, p in zip(ms, prev)])
        ok = np.isfinite(y)
        if ok.sum() >= 4:
            c = np.corrcoef(X[ok], y[ok])[0, 1]
            r2.append(c ** 2)
    cmp.append({"기간": pk, "기초일": b0d, "기말일": e1d, "시장지수수익률": mkt, "최종묶음가격수익률_평균": bp,
                "차이(묶음−시장,%p)": (bp - mkt) * 100, "±3%p이내": abs(bp - mkt) <= 0.03,
                "R2_중앙": np.median(r2), "R2_p25": np.percentile(r2, 25), "R2_p75": np.percentile(r2, 75),
                "회귀월수": len(ms), "R2계산묶음수": len(r2)})
T17 = pd.DataFrame(cmp)
verdict = ("묶음의 가격 손실은 시장 전체 하락을 따른 것이다" if T17["±3%p이내"].all() else "묶음 고유의 가격 요인이 있다")
IDXo = IDX.reset_index()
for c in ["월수익률", "가격보유곡비율", "수익률계산곡비율"]:
    IDXo[c] = (IDXo[c] * 100).round(3)
IDXo = IDXo.rename(columns={"월수익률": "월수익률(%)", "가격보유곡비율": "가격보유곡비율(%)", "수익률계산곡비율": "수익률계산곡비율(%)"})
IDXo["지수"] = IDXo["지수"].round(3)
tsave(IDXo, "3-17a_시장가격지수_월별")
T17o = T17.copy()
for c in ["시장지수수익률", "최종묶음가격수익률_평균"]:
    T17o[c] = (T17o[c] * 100).round(2)
T17o = T17o.rename(columns={"시장지수수익률": "시장지수수익률(%)", "최종묶음가격수익률_평균": "최종묶음가격수익률_평균(%)"}).round(3)
T17o["판정"] = verdict
tsave(T17o, "3-17_시장지수_대비_묶음가격")
log += ["## 2-2. 시장 가격지수 (표 3-17, 3-17a, 그림 3-11)\n",
        f"- cohort 767곡 동일가중 월말 지수({INDEX_START}=100). 매달 전월말·당월말 모두 30일 내 체결가가 있는 곡의 가격변화율 평균을 연결했다. "
        "가격은 '기준일 이전 30일 내 마지막 체결가'이므로 거래가 뜸한 곡은 가격이 늦게 반영된다.",
        "- R² = 묶음별 월 가격수익률(묶음 내 두 달 가격 보유 곡의 동일가중 평균)을 시장지수 월수익률에 회귀(단순회귀)한 결정계수. 묶음 곡도 지수에 포함되어 있다(767곡 중 20곡).",
        T17o.to_markdown(index=False), f"\n- **판단(사전 기준: 세 기간 모두 ±3%p 이내): {verdict}.**\n"]
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
xs = pd.to_datetime(IDX.index)
ax.plot(xs, IDX["지수"], color=BLUE, lw=2, label="동일가중 가격지수 (2023-10=100)")
ax.text(xs[-1], IDX["지수"].iloc[-1], f" {IDX['지수'].iloc[-1]:.1f}", va="center", fontsize=8, color=INK)
for pk, col in zip(PKS, ["#e6e5e0", "#f3f2ee", "#e6e5e0"]):
    ax.axvspan(PHASES[pk]["base_date"], PHASES[pk]["end_date"], color=col, alpha=0.6, lw=0)
    ax.text(PHASES[pk]["base_date"] + pd.Timedelta(days=15), 0.97, {"P1": "2024 평가", "P2": "2025 평가", "P2b": "2026.1~8 평가"}[pk],
            fontsize=8, color=INK, va="top", transform=ax.get_xaxis_transform())
ax.set_ylabel("지수 (2023-10 = 100)"); ax.set_title("cohort 767곡 동일가중 월말 가격지수 (매월 연결)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle(f"그림 3-11. 시장 가격지수 [{TAG}]", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.text(0.01, -0.04, f"주: 전월말·당월말 모두 30일 내 체결가가 있는 곡(월 {int(IDX['수익률계산곡수'].min())}~{int(IDX['수익률계산곡수'].max())}곡)의 동일가중 평균 변화율을 연결. "
         "음영 = 평가기간. 커버리지는 표 3-17a.", fontsize=7.5, color="#5f5e5a")
fig.tight_layout(); fig.savefig(FG / "3-11_시장가격지수.png", bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)

# ---- 사후 점검(판정 불변): 같은 기초·기말일의 보유형 시장 수익률과 비교
bh = []
for pk in PKS:
    P = PhaseData(pk); s, o = P.sel, P.out
    allr = (o["end_price_30"] / s["base_price"] - 1).dropna()
    el = apply_A(s, FINAL["A"]); a3 = (o.loc[el, "end_price_30"] / s.loc[el, "base_price"] - 1).dropna()
    bp = keep_bundles[(pk, "최종 묶음")][2]["price"].mean()
    mk = T17.set_index("기간").loc[pk, "시장지수수익률"]
    bh.append({"기간": pk, "최종묶음가격수익률_평균(%)": bp * 100, "시장지수_월연결(%)": mk * 100,
               "시장_보유형_전체적격곡평균(%)": allr.mean() * 100, "보유형곡수": len(allr),
               "A3후보_보유형평균(%)": a3.mean() * 100, "묶음−보유형시장(%p)": (bp - allr.mean()) * 100,
               "월연결지수−보유형시장(%p)": (mk - allr.mean()) * 100})
T17b = pd.DataFrame(bh).round(2)
tsave(T17b, "3-17b_사후점검_보유형시장비교")
log += ["### 사후 점검 (판정은 바꾸지 않음): 지수 구성 방식의 영향 (표 3-17b)\n",
        "- 같은 기초일에 사서 기말일까지 보유한 cohort 적격 곡 전체의 동일가중 평균 가격수익률(보유형 시장)과 비교했다.",
        T17b.to_markdown(index=False),
        f"\n- 묶음 가격수익률은 보유형 시장보다 세 기간 모두 {T17b['묶음−보유형시장(%p)'].min():.1f}~{T17b['묶음−보유형시장(%p)'].max():.1f}%p 높다(덜 하락). "
        f"사전 기준 판정과 달라지는 이유는 매월 연결 지수가 보유형 수익률보다 {T17b['월연결지수−보유형시장(%p)'].min():.1f}~{T17b['월연결지수−보유형시장(%p)'].max():.1f}%p 높게 나오기 때문이다. 거래가 뜸한 곡의 '30일 내 마지막 체결가'는 월별로 출렁이는데, "
        "월 산술평균 수익률을 연결하면 이 출렁임이 위쪽 편향(재조정 효과)으로 쌓인다. 따라서 사전 기준 판정('묶음 고유의 가격 요인이 있다')은 "
        "묶음의 문제라기보다 지수 산식의 차이를 반영한 결과로 해석해야 한다.\n"]

# ============================================================ 2-3·2-5 장기 보유, 현금 기준 누적 지급
lrows, crows = [], []
for key in long_phases():
    P = PhaseData(key)
    form = PHASES[key]["form"].date()
    r, (b, w, m) = run(key, FINAL, return_raw=True)
    ids = np.asarray(apply_A(P.sel, FINAL["A"])).reshape(-1, 1)
    ms = bundle_metrics(P, ids, np.ones_like(ids, dtype=float))
    nmon = len(PHASES[key]["eval"])
    for lab, mm in [("최종 묶음", m), ("단일곡(최종 적격기준)", ms)]:
        tr = mm["total_return"].dropna()
        lrows.append({"구성일": form, "보유개월": nmon, "구분": lab, "적격곡수": int(P.sel["eligible"].sum()),
                      "A3적격곡수": len(ids), "집계수": len(tr), "누적총수익률_평균": tr.mean(), "누적총수익률_중앙": tr.median(),
                      "누적총수익률_p5": tr.quantile(.05), "손실확률": (tr < 0).mean(),
                      "기초가격커버리지(cohort중)": P.sel["base_price"].notna().mean()})
        y = mm["yield"]
        crows.append({"구성일": form, "보유개월": nmon, "구분": lab, "표본": len(y), "누적지급÷원금_평균": y.mean(),
                      "누적지급÷원금_p5": y.quantile(.05), "누적지급÷원금_중앙": y.median(),
                      "연환산_평균": y.mean() * 12 / nmon})
T18, T20 = pd.DataFrame(lrows), pd.DataFrame(crows)
for T, cs in [(T18, ["누적총수익률_평균", "누적총수익률_중앙", "누적총수익률_p5", "손실확률", "기초가격커버리지(cohort중)"]),
              (T20, ["누적지급÷원금_평균", "누적지급÷원금_p5", "누적지급÷원금_중앙", "연환산_평균"])]:
    T[cs] = (T[cs] * 100).round(2)
    T.rename(columns={c: c + "(%)" for c in cs}, inplace=True)
tsave(T18, "3-18_장기보유성과")
tsave(T20, "3-20_현금기준_누적지급")
log += ["## 2-3. 장기 보유 성과 (표 3-18)\n",
        "- 구성일별 A3 적격 기준(구성 시점 정보)으로 최종 묶음 3,000개와 단일곡을 만들고 2026-08-31까지 보유. 누적 총수익률 = (누적 지급 + 종료일 가격 − 구성일 가격) ÷ 구성일 가격. "
        "선정 변수 규칙은 기존과 같다(2023-10-01: LTM 2022-08~2023-07, 비교 2021-08~2022-07, LTY 2020-10~2023-09 중 가용분, 기초가격 2023-09-30 이전 30일 내).",
        "- 2023-10-01 구성은 기초가격 30일 커버리지가 39%로 낮아 적격 표본이 작다.",
        T18.to_markdown(index=False), "", "## 2-5. 현금 기준 누적 지급 (표 3-20)\n",
        "- 가격을 무시하고 보유할 때 원금(구성일 가격) 대비 누적 지급. 묶음은 기말가격 조건 없이 20곡 전체로 계산.",
        T20.to_markdown(index=False), ""]

# ============================================================ 2-4 가격 하락 회복 기간
D = keep_bundles[("P2b", "최종 묶음")][2].dropna()
neg = D[D["price"] < 0]
yrs = -neg["price"] / (neg["pay"] * ANN["P2b"])
T19 = pd.DataFrame([{"기간": "P2b (2026.1~8)", "집계묶음수": len(D), "가격수익<0 묶음수": len(neg),
                     "가격하락_중앙(%)": neg["price"].median() * 100, "연환산지급수익률_중앙(%)": (neg["pay"] * ANN["P2b"]).median() * 100,
                     "회복연수_중앙": yrs.median(), "회복연수_p25": yrs.quantile(.25), "회복연수_p75": yrs.quantile(.75),
                     "회복연수_p95": yrs.quantile(.95)}]).round(3)
tsave(T19, "3-19_가격하락_회복기간")
log += ["## 2-4. 가격 하락 회복 기간 (표 3-19)\n",
        "- 2026년 1~8월 최종 묶음 중 가격수익률이 음수인 묶음에서, 회복 연수 = 가격 하락분 ÷ 연환산 지급수익률(같은 기말가격 확보 곡 기준, 8개월×12/8). "
        "지급이 현재 수준으로 유지되고 가격이 더 내리지 않는다는 단순 가정이다.",
        T19.to_markdown(index=False), ""]
(LG / "21_수익분해_보유기간.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
