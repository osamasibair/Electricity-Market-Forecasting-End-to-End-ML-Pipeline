"""Tune PatchTSTs settings with optuna, choosing on the calibration year only, so the test year stays unseen.
Each trial trains on the data before the calibration year with early stopping on it, and is scored by its best
calibration-year pinball loss. A trial whose loss is worse than the median of earlier trials at the same epoch is
stopped early (median pruning). The first trial is the current default settings, so the gain from tuning is visible.
The study is saved in models/optuna_patchtst.db, so it can be stopped with Ctrl+C and resumed by running this again,
and the best settings are written to models/patchtst_tuned.json for train_patchtst.py --tuned and ablation.py --tuned."""
import argparse
import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import optuna

from ablation import dates
from db import load_table
from evaluate import metrics
from forecast_windows import day_windows, windows
from tracking import log_run
from train_patchtst import models, rows, tuned_path

docs = Path("docs")
storage = f"sqlite:///{models / 'optuna_patchtst.db'}"
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
search = ["learning_rate", "weight_decay", "dropout", "d_model", "layers", "hidden", "patch_len", "batch_size", "lags"]


def suggest(trial):
    """The settings to try in this trial. Patches always overlap by half, as in the defaults."""
    patch_len = trial.suggest_categorical("patch_len", [12, 24, 48])
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "layers": trial.suggest_int("layers", 1, 3),
        "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
        "patch_len": patch_len,
        "stride": patch_len // 2,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "lags": trial.suggest_categorical("lags", [0, 1]),
    }


def objective(trial, df, train_starts, calib_days, actual_calib):
    from patchtst import PatchTSTForecaster

    def report(epoch, loss):
        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model = PatchTSTForecaster(**suggest(trial)).fit(df, train_starts, valid=calib_days, callback=report)
    median = model.predict(df, calib_days)[..., 1].ravel()
    trial.set_user_attr("calibration_mape", float(metrics(actual_calib, median)["mape"]))
    trial.set_user_attr("best_epoch", model.best_epochs)
    return min(model.valid_losses)


def best_settings(trial):
    return {**trial.params, "stride": trial.params["patch_len"] // 2}


def describe(trial):
    p = trial.params
    return (f"{trial.number:>5} {trial.value:>11.4f} {trial.user_attrs.get('calibration_mape', float('nan')):>6.2f}% "
            f"{trial.user_attrs.get('best_epoch', 0):>6} {p['learning_rate']:>8.1e} {p['dropout']:>7.2f} {p['d_model']:>7} "
            f"{p['layers']:>6} {p['hidden']:>6} {p['patch_len']:>5} {p['batch_size']:>5} {p['lags']:>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40, help="how many trials to run this time (default 40)")
    parser.add_argument("--timeout", type=float, help="stop starting new trials after this many minutes")
    args = parser.parse_args()

    from patchtst import defaults

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        calibration_start, split_date, target = pool.submit(dates).result()
    df = load_table("features")
    calib_days = day_windows(df.index, calibration_start, split_date)
    actual_calib = df[target].to_numpy()[rows(calib_days)]
    train_starts = windows(df.index, last=calibration_start, stride=defaults["train_stride"])

    models.mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name="patchtst", storage=storage, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    if not study.trials:
        study.enqueue_trial({name: defaults[name] for name in search})
    print(f"tuning on {len(train_starts):,} training windows, scored on {len(calib_days)} calibration days")
    try:
        study.optimize(lambda trial: objective(trial, df, train_starts, calib_days, actual_calib),
                       n_trials=args.trials, timeout=args.timeout * 60 if args.timeout else None)
    except KeyboardInterrupt:
        print("\nstopped, keeping the trials finished so far (run again to continue)")

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        raise SystemExit("no finished trials yet")
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"\n{len(done)} finished trials, {pruned} stopped early")
    print(f"{'trial':>5} {'calib loss':>11} {'MAPE':>7} {'epoch':>6} {'lr':>8} {'dropout':>7} {'d_model':>7} "
          f"{'layers':>6} {'hidden':>6} {'patch':>5} {'batch':>5} {'lags':>4}")
    if study.trials[0].state == optuna.trial.TrialState.COMPLETE:
        print(describe(study.trials[0]) + "   <- defaults")
    for trial in sorted(done, key=lambda t: t.value)[:5]:
        print(describe(trial))

    best = study.best_trial
    tuned_path.write_text(json.dumps(best_settings(best), indent=2) + "\n")
    print(f"\nbest: trial {best.number}, saved its settings to {tuned_path}")

    if len(done) >= 5:
        import matplotlib.pyplot as plt

        docs.mkdir(exist_ok=True)
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        ax.figure.set_size_inches(7, 4)
        ax.figure.tight_layout()
        ax.figure.savefig(docs / "patchtst_tuning.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("saved docs/patchtst_tuning.png")

    log_run(experiment="demand-patchtst-tuning", params=best_settings(best),
            metrics={"calibration_loss": best.value, "calibration_mape": best.user_attrs["calibration_mape"],
                     "default_calibration_loss": study.trials[0].value or float("nan"), "trials": len(done)},
            files=[tuned_path])