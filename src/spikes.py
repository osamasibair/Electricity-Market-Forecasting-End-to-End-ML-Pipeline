"""Classify half-hours where the price spikes well above its recent level."""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from db import load_table
from train import split_date

models = Path("models")
validation_start = "2024-09-01"


def label_spikes(df, threshold):
    return (df["price"] - df["price_roll_7d"] > threshold).astype(int)


def make_classifier():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)


if __name__ == "__main__":
    df = load_table("price_features")
    df["price_excess_recent"] = df["price_lag_recent"] - df["price_roll_7d"]
    base = joblib.load(models / "price_model.joblib")["features"]
    candidates = {
        "price features": base,
        "+ recent excess": base + ["price_excess_recent"],
    }
    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    threshold = (train["price"] - train["price_roll_7d"]).quantile(0.95)
    y_train = label_spikes(train, threshold)
    y_test = label_spikes(test, threshold)
    print(f"spike = price more than £{threshold:.2f}/MWh above its 7 day average")
    print(f"spike rate: train {y_train.mean():.1%}, test {y_test.mean():.1%}")

    fit_part = train[train.index < validation_start]
    val = train[train.index >= validation_start]
    y_fit = label_spikes(fit_part, threshold)
    y_val = label_spikes(val, threshold)
    val_ap = {}
    val_probs = {}
    for name, cols in candidates.items():
        val_model = make_classifier()
        val_model.fit(fit_part[cols], y_fit)
        val_probs[name] = val_model.predict_proba(val[cols])[:, 1]
        val_ap[name] = average_precision_score(y_val, val_probs[name])
        print(f"validation  {name:16} average precision {val_ap[name]:.3f}")
    best = max(val_ap, key=val_ap.get)
    features = candidates[best]
    print(f"chosen on validation: {best}")

    cutoffs = np.arange(0.05, 0.55, 0.05)
    f1s = [f1_score(y_val, (val_probs[best] >= c).astype(int)) for c in cutoffs]
    cutoff = cutoffs[int(np.argmax(f1s))]
    print(f"cut off chosen on validation: {cutoff:.2f}")

    model = make_classifier()
    model.fit(train[features], y_train)
    prob = model.predict_proba(test[features])[:, 1]
    pred = (prob >= 0.5).astype(int)
    tuned = (prob >= cutoff).astype(int)

    persistence = (test["price_excess_recent"] > threshold).astype(int)
    for name, p in {"persistence": persistence, "lightgbm 0.5": pred, "lightgbm tuned": tuned}.items():
        print(f"{name:15} precision {precision_score(y_test, p):.2f}  "
              f"recall {recall_score(y_test, p):.2f}  F1 {f1_score(y_test, p):.2f}")
    print(f"average precision: lightgbm {average_precision_score(y_test, prob):.3f}, "
          f"persistence {average_precision_score(y_test, persistence):.3f}, "
          f"random {y_test.mean():.3f}")
    print(f"ROC AUC {roc_auc_score(y_test, prob):.3f}")

    joblib.dump({"model": model, "features": features, "threshold": threshold, "cutoff": cutoff},
                models / "spike_model.joblib")
    print("saved models/spike_model.joblib")