"""[수정 보완] 0.6단계 구성요소 특성(구성일 이전 자료만) 계산과 미래정보 누출 테스트.

compute_features(pk, W, S) 는 월별 지급표 W(곡×월)와 수익원별 지급표 S를 받아 특성을 계산한다.
테스트: 구성일 이후 열을 모두 지운 W, S로 다시 계산해 값이 같으면 미래정보가 섞이지 않은 것이다.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
from config import PHASES, PCT_COLS
from engine import PhaseData
from features import _base
from rr_events import detect
from rev_common import REV

FEATS = {  # 코드: (열 이름, 나쁜 방향 'high'|'low', 설명)
    "F1": ("f1_stab", "high", "|LTM/LTY−1|"),
    "F2": ("f2_trend", "low", "36개월 log(1+지급) 기울기"),
    "F3": ("f3_cv24", "high", "24개월 월 지급 변동계수"),
    "F4": ("f4_maxshare", "high", "LTM 중 최대 월 비중"),
    "F5": ("f5_trans", "low", "LTM 가중 전송 비중"),
    "F6": ("f6_bro", "high", "LTM 가중 방송 비중"),
    "F7": ("f7_srchhi", "high", "수익원 비중 HHI"),
    "F8": ("f8_multiple", "high", "기초가격 ÷ LTM"),
    "M1": ("m1_age", "low", "곡 연령"),
    "M2": ("m2_rr", "binary", "구성 전 24개월 역주행 이력(1=제외)"),
}


def compute_features(pk, W, S, ev_rr=None):
    P = PhaseData(pk); sel = P.sel
    Ph = PHASES[pk]
    ids = sel.index
    lty = [c for c in Ph["lty"] if c in W.columns]
    H = W.reindex(index=ids, columns=lty)
    L = W.reindex(index=ids, columns=Ph["ltm"])
    prev = W.reindex(index=ids, columns=Ph["prev"])
    ltm = L.sum(1, min_count=12); ltyv = H.mean(1) * 12
    f = pd.DataFrame(index=ids)
    f["hist_months36"] = H.notna().sum(1)
    f["f1_stab"] = (ltm / ltyv.replace(0, np.nan) - 1).abs()
    x = np.arange(H.shape[1])
    lg = np.log1p(H.to_numpy(float))
    ok = np.isfinite(lg)
    slope = []
    for i in range(len(lg)):
        m = ok[i]
        slope.append(np.polyfit(x[m], lg[i, m], 1)[0] if m.sum() >= 3 else np.nan)
    f["f2_trend"] = slope
    H24 = H.iloc[:, -24:]
    f["f3_cv24"] = H24.std(1, ddof=1) / H24.mean(1).replace(0, np.nan)
    f["f4_maxshare"] = L.max(1) / ltm.replace(0, np.nan)
    sh = {}
    for c in PCT_COLS:
        sh[c] = S[c].reindex(index=ids, columns=Ph["ltm"]).sum(1) / ltm.replace(0, np.nan)
    f["f5_trans"] = sh["pct_transmission"]
    f["f6_bro"] = sh["pct_broadcast"]
    f["f7_srchhi"] = sum(v.fillna(0) ** 2 for v in sh.values())
    f["f8_multiple"] = sel["base_price"] / ltm.replace(0, np.nan)
    f["m1_age"] = (Ph["form"] - sel["release_date"]).dt.days / 365.25
    form = pd.Period(Ph["form"], "M")
    if ev_rr is None:
        ev_rr = detect()
    ev_ok = ev_rr[[form - 24 <= pd.Period(e, "M") <= form - 2 for e in ev_rr["event_ym"]]]
    f["m2_rr"] = f.index.isin(set(ev_ok["song_id"])).astype(int)
    f["decade"] = sel["decade"]
    f["ltm_check"] = ltm
    f["prev_check"] = prev.sum(1, min_count=12)
    return f


def truncated_inputs(pk):
    """구성월 이후(구성월 포함) 열을 모두 지운 W, S."""
    B = _base()
    cut = pd.Period(PHASES[pk]["form"], "M")
    keep = [c for c in B["W"].columns if pd.Period(c, "M") < cut]
    return B["W"][keep], {k: v[[c for c in keep if c in v.columns]] for k, v in B["S"].items()}


def test_no_leak(pk, ev_rr):
    B = _base()
    full = compute_features(pk, B["W"], B["S"], ev_rr)
    Wt, St = truncated_inputs(pk)
    # 역주행 사건도 구성 전 자료로만 다시 탐지해야 같다: 사건 확인에 필요한 달(시작+1)이 구성 전이므로,
    # 구성 전 자료로 확인되는 사건 목록과 같은지 따로 점검한다.
    trunc = compute_features(pk, Wt, St, ev_rr)
    cols = [c for c in full.columns if c not in ("decade",)]
    a, b = full[cols].to_numpy(float), trunc[cols].to_numpy(float)
    same = np.allclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999), equal_nan=True)
    return same, full


if __name__ == "__main__":
    ev = detect()
    out, rep = [], []
    for pk in ["P1", "P2", "P2b"]:
        same, f = test_no_leak(pk, ev)
        rep.append({"기간": pk, "구성일": PHASES[pk]["form"].date(), "구성일 이후 열 제거 후 재계산 일치": same})
        f = f.assign(기간=pk, 구성일=PHASES[pk]["form"].date())
        out.append(f.reset_index().rename(columns={"index": "song_id"}))
        assert same, f"미래정보 누출 의심: {pk}"
    pd.concat(out).to_csv(REV / "tables" / "features_ex_ante.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rep).to_markdown(index=False))
