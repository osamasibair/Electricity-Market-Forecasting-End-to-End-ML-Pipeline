"""Metrics for comparing forecasts."""
import numpy as np


def metrics(actual, predicted):
    error = actual - predicted
    mae = np.abs(error).mean()
    rmse = np.sqrt((error ** 2).mean()) #root mean square error
    mape = (np.abs(error) / actual).mean() * 100 #mean absolute percentage error
    return {"mae": mae, "rmse": rmse, "mape": mape}

def quantile_loss(actual, predicted, q):
    error = actual - predicted
    return np.maximum(q * error, (q - 1) * error).mean()