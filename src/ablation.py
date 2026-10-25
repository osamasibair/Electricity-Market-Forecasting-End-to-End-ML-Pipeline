"""Train PatchTST once for each value of one setting, everything else unchanged, and compare them on the same
calibration and test days. Each run is trained, early stopped and calibrated exactly as in train_patchtst.py. Results go to the transformer_ablation
table (one row per setting and value) and to the MLflow experiment demand-patchtst-ablation."""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from db import load_table, save_table
from forecast_windows import day_windows
from tracking import log_run
from train_patchtst import patchtst, rows


def dates():
    """The calibration and test dates. Read in a separate process because the modules that hold them import LightGBM."""
    from quantiles import calibration_start
    from train import split_date, target

    return calibration_start, split_date, target


def parse(value):
    for kind in (int, float):
        try:
            return kind(value)
        except ValueError:
            pass
    return value


def merge(old, new):
    """Replace the rows for the setting just tested and keep the rows for other settings."""
    if old is None:
        return new
    old = old[old["setting"] != new["setting"].iloc[0]]
    return pd.concat([old, new], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setting", help="the PatchTST setting to vary, e.g. pooling")
    parser.add_argument("values", nargs="+", help="the values to try")
    args = parser.parse_args()

    from patchtst import defaults

    if args.setting not in defaults:
        raise SystemExit(f"unknown setting {args.setting!r}, choose from {sorted(defaults)}")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        calibration_start, split_date, target = pool.submit(dates).result()

    df = load_table("features")
    calib_days = day_windows(df.index, calibration_start, split_date)
    test_days = day_windows(df.index, split_date)
    actual_calib = df[target].to_numpy()[rows(calib_days)]
    actual_test = df[target].to_numpy()[rows(test_days)]

    results = []
    for value in map(parse, args.values):
        print(f"\n{args.setting} = {value}")
        settings = {args.setting: value}
        result, model, adjustment, losses, _, _ = patchtst(
            df, calib_days, test_days, actual_calib, actual_test, calibration_start, split_date, settings)
        row = {"setting": args.setting, "value": str(value),
               "parameters": sum(p.numel() for p in model.net.parameters()),
               "best_epoch": model.best_epochs, "calibration_loss": min(losses), **result}
        results.append(row)
        log_run(experiment="demand-patchtst-ablation", params={**model.settings, "best_epochs": model.best_epochs},
                metrics={k: v for k, v in row.items() if k not in ("setting", "value")}, files=[])

    table = pd.DataFrame(results)
    print(f"\n{args.setting:>12} {'parameters':>11} {'epoch':>6} {'calib loss':>11} {'MAPE':>7} {'in range':>9} {'width':>8}")
    for _, r in table.iterrows():
        print(f"{r['value']:>12} {r['parameters']:>11,} {r['best_epoch']:>6} {r['calibration_loss']:>11.4f} "
              f"{r['mape']:>6.2f}% {r['coverage']:>8.1f}% {r['width']:>5.0f} MW")

    try:
        old = load_table("transformer_ablation").reset_index(drop=True)
    except ValueError:
        old = None
    save_table(merge(old, table), "transformer_ablation")
    print("saved transformer_ablation")