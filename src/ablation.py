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
from train_patchtst import base_settings, patchtst, rows


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


def run_settings(base, setting, value, seed):
    """The settings for one run: the base settings with one setting changed. Patches always overlap by half."""
    settings = {**base, setting: value, "seed": seed}
    if setting == "patch_len":
        settings["stride"] = value // 2
    return settings


def average(runs):
    """One row from several seeds: the mean of every number, plus the spread of the test MAPE between seeds."""
    frame = pd.DataFrame(runs)
    row = frame.mean(numeric_only=True).to_dict()
    row["mape_std"] = float(frame["mape"].std()) if len(frame) > 1 else 0.0
    row["seeds"] = len(frame)
    return row


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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="train once per seed and average, e.g. --seeds 42 1 2")
    parser.add_argument("--tuned", action="store_true", help="start from the settings in models/patchtst_tuned.json")
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

    base = base_settings(args.tuned)
    results = []
    for value in map(parse, args.values):
        runs = []
        for seed in args.seeds:
            print(f"\n{args.setting} = {value}, seed {seed}")
            settings = run_settings(base, args.setting, value, seed)
            result, model, adjustment, losses, _, _ = patchtst(
                df, calib_days, test_days, actual_calib, actual_test, calibration_start, split_date, settings)
            run = {"parameters": sum(p.numel() for p in model.net.parameters()),
                   "best_epoch": model.best_epochs, "calibration_loss": min(losses), **result}
            runs.append(run)
            log_run(experiment="demand-patchtst-ablation", params={**model.settings, "best_epochs": model.best_epochs},
                    metrics=run, files=[])
        results.append({"setting": args.setting, "value": str(value), "tuned": args.tuned, **average(runs)})

    table = pd.DataFrame(results)
    print(f"\n{args.setting:>12} {'parameters':>11} {'epoch':>6} {'calib loss':>11} {'MAPE':>15} {'in range':>9} {'width':>8}"
          f"   ({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''}{', tuned' if args.tuned else ''})")
    for _, r in table.iterrows():
        print(f"{r['value']:>12} {r['parameters']:>11,.0f} {r['best_epoch']:>6.1f} {r['calibration_loss']:>11.4f} "
              f"{r['mape']:>6.2f}% ± {r['mape_std']:.2f}  {r['coverage']:>8.1f}% {r['width']:>5.0f} MW")

    try:
        old = load_table("transformer_ablation").reset_index(drop=True)
    except ValueError:
        old = None
    save_table(merge(old, table), "transformer_ablation")
    print("saved transformer_ablation")