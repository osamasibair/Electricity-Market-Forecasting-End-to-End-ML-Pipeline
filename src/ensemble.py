"""Blend the LightGBM and PatchTST forecasts, choosing the blend weight on the calibration year and scoring it on the test year."""
import numpy as np
import pandas as pd

from db import load_table, save_table
from evaluate import metrics
from tracking import log_run

weights = np.round(np.arange(0, 1.0001, 0.05), 2)


def conformal_adjustment(actual, low, high, coverage=0.8):
    return float(np.quantile(np.maximum(low - actual, actual - high), coverage))


def blend(frame, weight):
    """The weighted average of the two models middle forecasts and of their ranges."""
    return pd.DataFrame({
        f"ensemble{part}": weight * frame[f"patchtst{part}"] + (1 - weight) * frame[f"lightgbm{part}"]
        for part in ("", "_low", "_high")
    }, index=frame.index)


def weight_table(calib, test):
    """For every weight: the error on the calibration year (used to choose it) and the error, coverage and width on the test year."""
    rows = []
    for weight in weights:
        c, t = blend(calib, weight), blend(test, weight)
        adjustment = conformal_adjustment(calib["actual"], c["ensemble_low"], c["ensemble_high"])
        low, high = t["ensemble_low"] - adjustment, t["ensemble_high"] + adjustment
        rows.append({
            "weight": weight,
            "calibration_mape": metrics(calib["actual"], c["ensemble"])["mape"],
            "test_mape": metrics(test["actual"], t["ensemble"])["mape"],
            "test_mae": metrics(test["actual"], t["ensemble"])["mae"],
            "adjustment": adjustment,
            "test_coverage": ((test["actual"] >= low) & (test["actual"] <= high)).mean() * 100,
            "test_width": (high - low).mean(),
        })
    return pd.DataFrame(rows)


def best_weight(table):
    return float(table.loc[table["calibration_mape"].idxmin(), "weight"])


if __name__ == "__main__":
    calib, test = load_table("transformer_calibration"), load_table("transformer_forecasts")
    table = weight_table(calib, test)
    chosen = best_weight(table)
    by_weight = table.set_index("weight")

    print(f"{'PatchTST share':>15} {'calibration MAPE':>17} {'test MAPE':>10} {'in range':>9} {'width':>9}")
    for weight, row in by_weight.iterrows():
        marker = "  <- chosen on the calibration year" if weight == chosen else ""
        print(f"{weight:15.2f} {row['calibration_mape']:16.2f}% {row['test_mape']:9.2f}% {row['test_coverage']:8.1f}% "
              f"{row['test_width']:6.0f} MW{marker}")

    lightgbm, patchtst, ensemble = by_weight.loc[0.0], by_weight.loc[1.0], by_weight.loc[chosen]
    print(f"\nchosen blend: {1 - chosen:.0%} LightGBM + {chosen:.0%} PatchTST")
    print(f"test MAPE: LightGBM {lightgbm['test_mape']:.2f}%, PatchTST {patchtst['test_mape']:.2f}%, blend {ensemble['test_mape']:.2f}%")

    save_table(table, "ensemble_weights")
    print("saved ensemble_weights for the dashboard")

    log_run(
        experiment="demand-ensemble",
        params={"patchtst_weight": chosen, "weights_tried": len(weights)},
        metrics={"ensemble_test_mape": ensemble["test_mape"], "lightgbm_test_mape": lightgbm["test_mape"],
                 "patchtst_test_mape": patchtst["test_mape"], "ensemble_test_coverage": ensemble["test_coverage"],
                 "ensemble_test_width": ensemble["test_width"]},
        files=[],
    )