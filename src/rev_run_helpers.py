"""[수정 보완] rev_run.py의 0.6단계 표본 구성 함수(재실행 없이 다른 스크립트에서 재사용)."""
import pandas as pd
from rev_common import REV
from rev_fast import Uni
from rev_features import FEATS

_F = None


def build06(pk):
    global _F
    if _F is None:
        f = pd.read_csv(REV / "tables" / "features_ex_ante.csv", encoding="utf-8-sig")
        _F = {k: g.set_index("song_id") for k, g in f.groupby("기간")}
    U = Uni.build(pk)
    f = _F[pk].loc[U.df["song_id"].to_numpy()]
    for c in [v[0] for v in FEATS.values()] + ["hist_months36"]:
        U.df[c] = f[c].to_numpy()
    U.df["decade"] = f["decade"].to_numpy()
    return U.subset(U.df["hist_months36"].to_numpy() >= 36)
