"""[보조·사후 탐색] 역주행 분석 1-2의 1~6 (추가작업 2단계).

기존 결과 파일과 frozen_strategy.json은 읽기만 한다. 결과는 results/extra/ 아래에 저장한다.
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from config import RESULTS, PHASES, CACHE
from data_prep import load_trades, last_trade
from features import decade_label
from engine import run, apply_A, PhaseData
from rr_events import detect, eligible_song_months, transmission_wide, EXTRA

warnings.filterwarnings("ignore")
TAG = "보조·사후 탐색"
TB = EXTRA / "tables"
fz = json.loads((RESULTS / "frozen_strategy.json").read_text(encoding="utf-8"))
FINAL = fz["final"]["spec"]
PKS = ["P1", "P2", "P2b"]
PH_YEAR = {"P1": 2024, "P2": 2025, "P2b": 2026}
log = [f"# [{TAG}] 추가작업 2단계: 역주행 분석\n",
       "- 사건 정의는 기본 정의(전송 지급 ≥ 직전 12개월 평균×3, 2개월 연속, 발매 3년 이상, 기준선 하위 10% 제외, 12개월 내 재발 병합)를 쓴다.",
       "- 기간·가격 규칙(기준일 당일 포함 이전 30일 내 마지막 체결)·지표 정의는 기존 분석과 같다.\n"]


def tsave(df, name):
    d = df.copy()
    d.insert(0, "표시", TAG)
    d.to_csv(TB / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


ev = detect()
T = transmission_wide()
cols = list(T.columns)
per = pd.PeriodIndex(cols, freq="M")
pos = {c: i for i, c in enumerate(cols)}
meta = PhaseData("P2").sel  # 곡 메타(가수·발매일 동일)
rel = pd.read_csv(RESULTS.parent / "data/raw/original/03_metadata/cohort_767.csv", encoding="utf-8-sig",
                  usecols=["song_id", "release_date"]).set_index("song_id")["release_date"].pipe(pd.to_datetime)

# ============================================================ 1. 빈도
el = eligible_song_months()
months_ts = per.to_timestamp()
age_mat = (months_ts.to_numpy()[None, :] - rel.loc[el.index].to_numpy()[:, None]) / np.timedelta64(1, "D") / 365.25
age_bin_mat = np.select([age_mat < 7, age_mat < 15], ["<7년", "7~15년"], "15년+")
dec_song = rel.loc[el.index].dt.year.map(decade_label)
ev["age_bin"] = pd.cut(ev["age_at_event"], [-np.inf, 7, 15, np.inf], right=False, labels=["<7년", "7~15년", "15년+"]).astype(str)
ev["decade"] = ev["release_year"].map(decade_label)
fr = []
# 연도별
for y in sorted(set(per.year)):
    m = el.loc[:, [c for c in cols if c.startswith(str(y))]]
    sm = int(m.to_numpy().sum())
    if sm == 0:
        continue
    n_songs = int(m.any(axis=1).sum())
    e = ev[ev["event_year"] == y]
    fr.append({"구분": "연도", "범주": str(y), "사건수": len(e), "사건곡수": e["song_id"].nunique(), "판정가능곡수": n_songs,
               "사건곡비율(%)": e["song_id"].nunique() / n_songs * 100, "판정가능곡월": sm,
               "100곡년당사건수": len(e) / (sm / 12) * 100})
# 연령 구간 (사건 시점 연령, 분모는 해당 연령 구간의 판정가능 곡-월)
for k in ["<7년", "7~15년", "15년+"]:
    msk = el.to_numpy() & (age_bin_mat == k)
    sm = int(msk.sum()); e = ev[ev["age_bin"] == k]
    fr.append({"구분": "사건시 곡연령", "범주": k, "사건수": len(e), "사건곡수": e["song_id"].nunique(),
               "판정가능곡수": int(msk.any(axis=1).sum()), "사건곡비율(%)": e["song_id"].nunique() / max(1, msk.any(axis=1).sum()) * 100,
               "판정가능곡월": sm, "100곡년당사건수": len(e) / (sm / 12) * 100 if sm else np.nan})
for k in ["1990년대 이전", "1990년대", "2000년대", "2010년대", "2020년대"]:
    rows_k = (dec_song == k).to_numpy()
    msk = el.to_numpy() & rows_k[:, None]
    sm = int(msk.sum()); e = ev[ev["decade"] == k]
    n_s = int(msk.any(axis=1).sum())
    fr.append({"구분": "발매연대", "범주": k, "사건수": len(e), "사건곡수": e["song_id"].nunique(), "판정가능곡수": n_s,
               "사건곡비율(%)": e["song_id"].nunique() / n_s * 100 if n_s else np.nan, "판정가능곡월": sm,
               "100곡년당사건수": len(e) / (sm / 12) * 100 if sm else np.nan})
F1 = pd.DataFrame(fr).round(3)
tsave(F1, "3-11_역주행_빈도")
log += ["## 1. 빈도\n", "- 분모 = 판정 가능 곡-월(기준선 유효·하위 10% 초과·연령 조건 충족, 다음 달 자료 존재). 100곡년당 사건수 = 사건수 ÷ (판정가능 곡-월/12) × 100.",
        "- 연도는 사건 시작월 기준. 2022년은 기준선(직전 12개월) 때문에 판정 가능 기간이 2022-01부터이고, 2026년은 2026-07까지다.",
        F1.to_markdown(index=False), ""]

# ============================================================ 2. 지속성
offsets = list(range(-12, 25))
Tv = T.to_numpy()
sid_row = {s: i for i, s in enumerate(T.index)}
path_rows, pers = [], []
for _, e in ev.iterrows():
    i, j, b = sid_row[e["song_id"]], pos[e["event_ym"]], e["baseline_12m"]
    for k in offsets:
        jj = j + k
        if 0 <= jj < len(cols):
            v1 = Tv[i, jj] / b
            v3 = np.nanmean(Tv[i, max(0, jj - 2):jj + 1]) / b if jj - 2 >= 0 else np.nan
            path_rows.append({"song_id": e["song_id"], "k": k, "ratio_1m": v1, "ratio_3m": v3})
P = pd.DataFrame(path_rows)
P.to_pickle(CACHE / "rr_paths.pkl")
for k in [0, 1, 3, 6, 12, 24]:
    g = P[P["k"] == k]
    pers.append({"사건후개월": k, "가용사건수": len(g),
                 "3개월평균비율_중앙": g["ratio_3m"].median(), "3개월평균비율_p25": g["ratio_3m"].quantile(.25),
                 "3개월평균비율_p75": g["ratio_3m"].quantile(.75),
                 "1개월비율_중앙": g["ratio_1m"].median(), "1개월비율_p25": g["ratio_1m"].quantile(.25), "1개월비율_p75": g["ratio_1m"].quantile(.75),
                 "1.5배이상유지비율(%,3개월평균)": (g["ratio_3m"] >= 1.5).mean() * 100 if len(g) else np.nan,
                 "사건전수준(1배)이하비율(%,3개월평균)": (g["ratio_3m"] <= 1.0).mean() * 100 if len(g) else np.nan})
F2 = pd.DataFrame(pers).round(3)
tsave(F2, "3-12_역주행_지속성")
g12 = P[P["k"] == 12]
log += ["## 2. 지속성(되돌림)\n",
        "- 비율 = 전송 지급 ÷ 사건 전 12개월 평균(기준선). 정산 몰림의 잡음을 줄이려고 주지표는 해당 시점까지의 최근 3개월 평균(k−2..k)을 쓰고, 1개월 값도 함께 보고한다.",
        "- 자료가 2026-08에서 끝나므로 늦게 시작한 사건은 +12·+24개월 값이 없다(가용사건수 참조).",
        F2.to_markdown(index=False),
        f"\n- 사건 후 12개월 시점에 사건 전 수준의 1.5배 이상을 유지한 사건: {(g12['ratio_3m'] >= 1.5).sum()}/{len(g12)}건 "
        f"({(g12['ratio_3m'] >= 1.5).mean():.1%}).\n"]

# ============================================================ 3. 가격 반응
t = load_trades()
W = PhaseData("P2")  # 지급 행렬 접근용
from features import _base
Wpay = _base()["W"]
ev_months = {s: [pd.Period(x, "M") for x in g["event_ym"]] for s, g in ev.groupby("song_id")}
pr_rows, pr_cmp = [], []
for _, e in ev.iterrows():
    p0 = pd.Period(e["event_ym"], "M")
    entry = (p0 + 2).to_timestamp(how="end").normalize()
    exit_ = (p0 + 14).to_timestamp(how="end").normalize()
    pre = (p0 - 1).to_timestamp(how="end").normalize()
    row = {"song_id": e["song_id"], "title": e["title"], "event_ym": e["event_ym"], "매수일": entry.date(), "매도일": exit_.date()}
    feasible = exit_ <= pd.Timestamp("2026-08-31") and entry >= pd.Timestamp("2023-09-19")
    row["분석가능기간"] = feasible
    if not feasible:
        pr_rows.append(row); continue
    a = last_trade(t, entry, 30); b = last_trade(t, exit_, 30); c = last_trade(t, pre, 30)
    pay_m = [str(p0 + k) for k in range(3, 15)]
    pay = Wpay.loc[:, [m for m in pay_m if m in Wpay.columns]].sum(axis=1)
    sid = e["song_id"]
    row["사건전가격(시작−1월말)"] = c["price"].get(sid, np.nan)
    row["매수가"] = a["price"].get(sid, np.nan); row["매도가"] = b["price"].get(sid, np.nan)
    row["사건전→매수 가격변화"] = row["매수가"] / row["사건전가격(시작−1월말)"] - 1
    row["가격수익률_12m"] = row["매도가"] / row["매수가"] - 1
    row["총수익률_12m"] = (row["매도가"] - row["매수가"] + pay.get(sid, np.nan)) / row["매수가"]
    # 같은 시점 비사건 곡 (매수일 기준 ±12개월 안에 사건이 없는 곡)
    ok = a.index.intersection(b.index)
    non = [s for s in ok if s != sid and not any(abs((m - (p0 + 2)).n) <= 12 for m in ev_months.get(s, []))]
    pr_non = b.loc[non, "price"] / a.loc[non, "price"] - 1
    tr_non = (b.loc[non, "price"] - a.loc[non, "price"] + pay.reindex(non)) / a.loc[non, "price"]
    row["비사건곡수"] = len(non)
    row["비사건_가격수익률_평균"] = pr_non.mean(); row["비사건_총수익률_평균"] = tr_non.mean()
    row["비사건_가격수익률_중앙"] = pr_non.median(); row["비사건_총수익률_중앙"] = tr_non.median()
    row["초과_가격수익률(사건−비사건평균)"] = row["가격수익률_12m"] - row["비사건_가격수익률_평균"]
    row["초과_총수익률(사건−비사건평균)"] = row["총수익률_12m"] - row["비사건_총수익률_평균"]
    pr_rows.append(row)
F3 = pd.DataFrame(pr_rows)
feas = F3[F3["분석가능기간"]]
cov_n = feas["총수익률_12m"].notna().sum()
summ = pd.DataFrame([{"전체사건": len(F3), "분석가능기간사건(시작 2023-07~2025-06)": len(feas), "매수·매도가 확보(커버리지)": int(cov_n),
                      "커버리지(%)": cov_n / max(1, len(feas)) * 100,
                      "사건전→매수 가격변화_중앙(%)": feas["사건전→매수 가격변화"].median() * 100,
                      "사건곡_가격수익률_중앙(%)": feas["가격수익률_12m"].median() * 100,
                      "사건곡_총수익률_중앙(%)": feas["총수익률_12m"].median() * 100,
                      "비사건_총수익률_중앙의중앙(%)": feas["비사건_총수익률_중앙"].median() * 100,
                      "초과총수익률_중앙(%p)": feas["초과_총수익률(사건−비사건평균)"].median() * 100,
                      "초과총수익률>0 사건비율(%)": (feas["초과_총수익률(사건−비사건평균)"].dropna() > 0).mean() * 100}]).round(2)
tsave(F3.round(4), "3-12b_역주행_가격반응_사건별")
tsave(summ, "3-12c_역주행_가격반응_요약")
log += ["## 3. 가격 반응\n",
        "- 매수 = 사건 시작 +2개월 말(사건 확인 직후) 이전 30일 내 체결가, 매도 = 매수 12개월 뒤 월말 이전 30일 내 체결가. 총수익률에는 시작 +3~+14개월 1주당 지급을 더한다.",
        "- 비교군 = 같은 매수·매도일에 가격이 있고 매수 시점 ±12개월 안에 사건이 없는 곡. 가격 자료는 2023-09-19 이후, 평가 끝은 2026-08-31이므로 사건 시작 2023-07~2025-06만 분석 가능하다.",
        summ.to_markdown(index=False), "",
        feas[["title", "event_ym", "사건전→매수 가격변화", "가격수익률_12m", "총수익률_12m", "비사건_총수익률_평균", "초과_총수익률(사건−비사건평균)"]]
        .round(3).to_markdown(index=False), ""]

# ============================================================ 4·6. 묶음 기여도와 쏠림
contrib, conc = [], []
for pk in PKS:
    Pd = PhaseData(pk)
    evm = set(PHASES[pk]["eval"])
    ev_ids = set(ev.loc[ev["event_ym"].isin(evm), "song_id"])
    elig = apply_A(Pd.sel, FINAL["A"])
    pool_ev = [s for s in elig if s in ev_ids]
    n, k, N = len(elig), len(pool_ev), FINAL["N"]
    _, (bundles, w, m) = run(pk, FINAL, return_raw=True)
    nb = bundles.shape[0]
    flat = bundles.ravel()
    g = lambda s: s.reindex(flat).to_numpy(dtype=float).reshape(nb, N)
    p0 = g(Pd.sel["base_price"]); evp = g(Pd.out["eval_pay"]); pvp = g(Pd.out["eval_prev_pay"])
    is_ev = np.isin(bundles, list(ev_ids))
    has = is_ev.any(1)
    sh = w / p0
    pay_i = sh * evp
    ev_share = (pay_i * is_ev).sum(1) / pay_i.sum(1)
    # 사건 곡 제외 후 동일 금액 재배분
    keep = ~is_ev
    w2 = np.where(keep, 1.0, 0.0); w2 = w2 / w2.sum(1, keepdims=True)
    sh2 = w2 / p0
    y1 = pay_i.sum(1); y2 = (sh2 * evp).sum(1)
    d1 = np.clip(1 - y1 / (sh * pvp).sum(1), 0, None); d2 = np.clip(1 - y2 / (sh2 * pvp).sum(1), 0, None)
    top1 = pay_i.max(1) / pay_i.sum(1)
    analytic = 1 - np.exp(sum(np.log((n - k - i) / (n - i)) for i in range(N))) if n - k >= N else 1.0
    contrib.append({"기간": pk, "평가기간": f"{PHASES[pk]['eval'][0]}~{PHASES[pk]['eval'][-1]}",
                    "적격후보곡수(A3)": n, "후보중_평가기간사건곡수": k,
                    "단일곡_사건곡보유확률(%)": k / n * 100,
                    "묶음_1곡이상보유비율(%,모의)": has.mean() * 100, "묶음_1곡이상보유확률(%,해석해)": analytic * 100,
                    "묶음당사건곡수_평균": is_ev.sum(1).mean(),
                    "사건포함묶음_사건곡지급비중_평균(%)": ev_share[has].mean() * 100 if has.any() else np.nan,
                    "사건포함묶음_사건곡지급비중_중앙(%)": np.median(ev_share[has]) * 100 if has.any() else np.nan,
                    "사건포함묶음_수익률변화(제외−포함,%p)": (y2 - y1)[has].mean() * 100 if has.any() else np.nan,
                    "사건포함묶음_감소율변화(제외−포함,%p)": (d2 - d1)[has].mean() * 100 if has.any() else np.nan,
                    "전체묶음_수익률변화(%p)": (y2 - y1).mean() * 100, "전체묶음_감소율변화(%p)": (d2 - d1).mean() * 100,
                    "단일곡_사건곡평균수익률(%)": (Pd.out.loc[pool_ev, "eval_pay"] / Pd.sel.loc[pool_ev, "base_price"]).mean() * 100 if k else np.nan,
                    "단일곡_비사건곡평균수익률(%)": (Pd.out.loc[[s for s in elig if s not in ev_ids], "eval_pay"] /
                                              Pd.sel.loc[[s for s in elig if s not in ev_ids], "base_price"]).mean() * 100})
    for lab, msk in [("사건 포함 묶음", has), ("사건 없는 묶음", ~has)]:
        if msk.any():
            conc.append({"기간": pk, "구분": lab, "묶음수": int(msk.sum()),
                         "1위곡지급비중_중앙(%)": np.median(top1[msk]) * 100, "1위곡지급비중_p95(%)": np.percentile(top1[msk], 95) * 100,
                         "1위곡이사건곡인비율(%)": (np.take_along_axis(is_ev, pay_i.argmax(1)[:, None], 1)[:, 0][msk]).mean() * 100})
F4 = pd.DataFrame(contrib).round(3)
F6 = pd.DataFrame(conc).round(3)
tsave(F4, "3-13_역주행_묶음기여")
tsave(F6, "3-13b_역주행_쏠림")
log += ["## 4. 묶음 기여도 (최종 전략 A3·20곡·동일 금액 묶음 3,000개)\n",
        "- 사건 곡 = 사건 시작월이 해당 평가기간 안에 있는 곡. 사건 곡 제외는 나머지 곡을 동일 금액으로 재배분해 다시 계산했다.",
        "- 해석해 = 1 − C(n−k, 20)/C(n, 20) (n=적격 후보 수, k=후보 중 사건 곡 수).",
        F4.T.to_markdown(), "", "## 6. 쏠림 (평가기간 실제 지급 기준 1위 곡 비중)\n", F6.to_markdown(index=False), ""]

# ============================================================ 5. 과거 역주행 이력의 예측력
pr = []
for pk in PKS:
    Pd = PhaseData(pk); s = Pd.sel; o = Pd.out
    form = pd.Period(PHASES[pk]["form"], "M")
    lo, hi = form - 24, form - 2  # 시작월 + 1 ≤ 구성 전월 → 시작월 ≤ 구성월−2
    hist_ids = set(ev.loc[[lo <= pd.Period(x, "M") <= hi for x in ev["event_ym"]], "song_id"])
    d = s[s["eligible"]].copy()
    d["song_id"] = d.index
    d["rr_hist"] = d.index.isin(hist_ids).astype(int)
    ann = 12 / len(PHASES[pk]["eval"])
    d["yield_next"] = o.loc[d.index, "eval_pay"] / d["base_price"] * ann
    d["decline_next"] = (1 - o.loc[d.index, "eval_pay"] / o.loc[d.index, "eval_prev_pay"]).clip(lower=0)
    d["year"] = PH_YEAR[pk]
    d["a3_pass"] = (d["stab"] <= FINAL["A"]["stab_th"] + 1e-12).astype(int)
    pr.append(d)
R = pd.concat(pr, ignore_index=True)
R["artist"] = R["singer"].astype("category").cat.codes
R["trans_share"] = R["sh_transmission"].fillna(0)
reg = []
for yv in ["yield_next", "decline_next"]:
    f = f"{yv} ~ rr_hist + age + broadcast_share + trans_share + C(year)"
    m = smf.ols(f, data=R).fit(cov_type="cluster", cov_kwds={"groups": R["artist"]})
    reg.append({"종속변수": {"yield_next": "다음 해 지급수익률(연환산)", "decline_next": "다음 해 지급감소율"}[yv],
                "역주행이력_계수": m.params["rr_hist"], "군집표준오차": m.bse["rr_hist"], "p값": m.pvalues["rr_hist"],
                "관측치": int(m.nobs), "이력곡-연도수": int(R["rr_hist"].sum()), "군집수": R["artist"].nunique(), "R2": m.rsquared})
F5 = pd.DataFrame(reg).round(4)
desc = R.groupby(["year", "rr_hist"]).agg(곡수=("yield_next", "size"), 다음해수익률_평균=("yield_next", "mean"),
                                          다음해감소율_평균=("decline_next", "mean"), A3통과율=("a3_pass", "mean"),
                                          stab_중앙=("stab", "median")).reset_index()
tot = R.groupby("rr_hist").agg(곡수=("yield_next", "size"), 다음해수익률_평균=("yield_next", "mean"),
                               다음해감소율_평균=("decline_next", "mean"), A3통과율=("a3_pass", "mean"), stab_중앙=("stab", "median")).reset_index()
tot.insert(0, "year", "전체")
desc = pd.concat([desc, tot], ignore_index=True)
for c in ["다음해수익률_평균", "다음해감소율_평균", "A3통과율"]:
    desc[c] = (desc[c] * 100).round(2)
desc = desc.rename(columns={"rr_hist": "직전24개월역주행", "다음해수익률_평균": "다음해수익률_평균(%)",
                            "다음해감소율_평균": "다음해감소율_평균(%)", "A3통과율": "A3통과율(%)"})
tsave(F5, "3-14_역주행_예측력")
tsave(desc, "3-14b_역주행이력_기술통계_A3필터")
log += ["## 5. 과거 역주행 이력의 예측력\n",
        "- 표본: 각 구성일(2024·2025·2026-01-01)의 적격 곡(기초가격 30일 내). 이력 = 사건 시작월이 구성월 24개월 전 ~ 구성월 2개월 전(두 번째 달까지 구성 전에 관측). "
        "다음 해 지급수익률은 2026년(8개월)을 12/8로 연환산. 통제: 곡 연령, 방송 비중, 전송 비중, 연도 고정효과. 아티스트 군집 표준오차.",
        F5.to_markdown(index=False), "", "### 기술통계와 A3 안정성 필터 통과율\n", desc.to_markdown(index=False), ""]

(EXTRA / "logs" / "12_역주행_분석.md").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
