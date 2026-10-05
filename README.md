# UK Electricity Market Forecasting End to End ML Pipeline

[![CI](https://github.com/osamasibair/Electricity-Market-Forecasting-End-to-End-ML-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/osamasibair/Electricity-Market-Forecasting-End-to-End-ML-Pipeline/actions/workflows/tests.yml)

An end to end machine learning pipeline that forecasts Great Britain's half hourly electricity demand and price one day ahead, from raw public data to a tested, containerised API, and backtests a battery trading strategy on the price forecasts.

- **Demand: LightGBM day ahead model: 5.03% MAPE, about 40% lower error than a same-time-last-week baseline (8.40%), with 80% prediction intervals achieving 81.9% coverage.**
- **Price: LightGBM model with £21.50/MWh MAE, 15% lower than the best naive baseline, using wind, solar and net demand to forecast cheap and expensive periods.**
- **Spikes: LightGBM classifier that ranks spike risk 2.5x better than persistence (average precision 0.43 vs 0.17, ROC AUC 0.90).**
- **Backtest: a simulated 1 MW / 2 MWh battery scheduled from the price forecast earns 77% of the perfect foresight profit, against 52% for a baseline forecast and 74% for a simple average of the last 7 days prices.**

**Stack:** Python, pandas, LightGBM, FastAPI, Docker, pytest, SciPy, Ruff, GitHub Actions, PostgreSQL, SQLAlchemy, Docker Compose.

---

## Results

Test period: 1 September 2025 to 1 September 2026 (17,566 half-hours never seen in training).

### Demand forecast

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive baseline (same time last week) | 2,132 MW | 2,878 MW | 8.40% |
| **LightGBM** | **1,247 MW** | **1,628 MW** | **5.03%** |

| Day | MAPE |
|---|---|
| Typical weekday (26 August 2026) | 2.67% |
| Christmas Day 2025 | 13.52% |

### Demand prediction intervals

Three LightGBM quantile models (q10, q50, q90) give an 80% prediction interval around each demand forecast. A well calibrated 80% interval should contain about 80% of actual values, with about 10% falling below and 10% above.

| Intervals | Coverage | Below | Above | Average width |
|---|---|---|---|---|
| Raw quantile models | 51.3% | 20.7% | 28.0% | 2,466 MW |
| Conformally calibrated | 75.8% | 11.2% | 13.0% | 3,969 MW |
| **Calibrated, models retrained on full training period** | **81.9%** | **9.4%** | **9.1%** | **4,195 MW** |

The raw quantile models were overconfident, covering only half of actual values. Calibration widened each interval by 758 MW, bringing coverage to the 80% target with misses balanced on both sides. The intervals also adapt to the time of day: they are narrowest overnight (around 3,000 MW) and widest around midday (up to around 6,000 MW).

### Price forecast

The target is the Market Index Price, the half hourly price of electricity traded in the short term market. Prices can be negative, so models are scored on MAE and RMSE in £/MWh rather than MAPE.

| Model | MAE | RMSE |
|---|---|---|
| Naive baseline: same time last week | £27.57 | £41.25 |
| Naive baseline: most recent known day | £25.25 | £39.22 |
| **LightGBM** | **£21.50** | **£29.45** |

LightGBM is 15% better than the stronger baseline on MAE and 25% better on RMSE. The larger RMSE improvement means it mostly avoids the big misses the baselines make, by seeing windy, cheap periods and tight, expensive ones coming.

**Feature selection on a validation year.** The first price model relied heavily on `dayofyear`. Unlike demand, price levels drift between years with gas prices (around £150/MWh in early 2023, around £70 in 2024, rising again through 2026), so seasonal patterns learned from past years don't generalise. Comparing feature sets on a held-out validation year (September 2024 to August 2025), removing `dayofyear` improved MAE from £18.57 to £17.12 (7.8%). That version was chosen before the test set was evaluated, then tested once.

### Price spike classifier

A spike is defined relative to the recent price level: **a price more than £43.72/MWh above its own 7 day average**. The threshold was set so that 5% of training half-hours count as spikes, using training data only. The test year was more volatile, with 6.3% spikes.

The baseline is persistence: predict a spike if the most recent known price at that time was itself spike-level.

| Model | Precision | Recall | F1 | Average precision |
|---|---|---|---|---|
| Random guessing | — | — | — | 0.063 |
| Persistence | 0.39 | 0.33 | 0.36 | 0.170 |
| LightGBM, 0.5 cut-off | 0.75 | 0.12 | 0.21 | **0.428** |
| LightGBM, cut-off chosen on validation (0.30) | 0.59 | 0.19 | 0.29 | **0.428** |

ROC AUC: 0.899.

- **The classifier ranks risk far better than persistence.** Its average precision, which scores the probabilities across every possible cut off, is 2.5 times persistence's and nearly 7 times random guessing.
- **Its alarms are more precise:** 59% of half hours it flags are real spikes, against 39% for persistence.
- **At a single cut off it catches fewer spikes:** Persistence scores a slightly higher F1. The cut off was chosen on the validation year, and the more volatile test year needed a less cautious one. It was not retuned on the test set.
- In the battery backtest below, using the spike probabilities to adjust the price forecast added no profit. The classifier ranks risk well, but its signal doesn't change one cycle a day trading decisions.

**Feature choice:** Tree models split on one feature at a time, so they can't easily compute the difference between two features. Adding `price_excess_recent` (the most recent known price minus its 7 day average, persistence's own signal) raised validation average precision from 0.405 to 0.412 and was kept.

### Battery trading backtest

Does a more accurate forecast actually make more money? A simulated battery plans each day's charging and discharging from a price forecast, using linear programming to find the most profitable schedule, and the plan is then scored on the real prices.

**Battery:** 1 MW power, 2 MWh storage, 90% round-trip efficiency, at most one full cycle a day, each day starting empty.

| Strategy | Forecast used to plan | Profit over test year | Share of perfect foresight | Losing days |
|---|---|---|---|---|
| Perfect foresight | Real prices (upper limit) | £42,937 | 100.0% | 0 |
| LightGBM | Price model | **£33,095** | **77.1%** | 17 |
| LightGBM + spike model | Price model, raised to the spike level where spike probability passes the cut-off | £33,048 | 77.0% | 16 |
| 7-day average | Average price at the same half-hour over the last 7 known days | £31,872 | 74.2% | 24 |
| Recent day | Most recent known price at each half hour | £22,377 | 52.1% | 45 |

- **The forecast is worth real money:** about £10,700 a year more than the recent day forecast (+48%), with far fewer losing days.
- **Most of the value comes from the daily shape.** Simply averaging the last 7 days prices captures 74%. A battery only needs to know when prices are low and high, not their exact level, so LightGBM's accuracy advantage turns into only about £1,200 (4%) more profit than the 7 day average.
- **The spike model doesnt change the outcome.**
- **The LightGBM result is an upper bound.** The price model uses actual wind, solar and demand rather than forecasts, so with real inputs its small lead over the 7 day average could shrink. The 7 day average uses only past prices, so its result is fully realistic.

### What the data shows about prices

- **Daily shape:** an evening peak at 6–7pm UK time (about £112/MWh on average), a smaller morning peak, an overnight dip (about £69), and a midday dip (about £74) caused by solar.
- **Fat tails:** the middle half of prices sits between £68 and £104/MWh, but prices ranged from −£102.92 to £1,352.90.
- **Negative prices** occurred in 1,684 half hours (2.6%), clustered around midday from solar and overnight on windy nights with low demand, and concentrated between April and September. They almost never occur during the evening peak.

### Error analysis

- **Solar:** errors peaked in the middle of the day, at 2.0–2.4 GW around midday compared with about 1 GW overnight. Rooftop solar isn't measured in this demand data, so sunny days reduce the demand the grid sees. Adding shortwave radiation as a feature reduced MAPE from 5.66% to 5.04%.
- **Christmas:** Christmas Day had a MAPE of 14.8%, with only two Christmases in the training data. A Christmas-period flag reduced this to 13.5%, but the model still overestimates demand.
- **Peaks:** the model gets the shape of each day right but underestimates the extremes, forecasting too low at the morning ramp and the evening peak.

---

## Quick start

The data and trained model aren't stored in the repository, so the pipeline has to be run once to create them.

**1. Set up and run the pipeline**

The pipeline stores its processed data in PostgreSQL, which runs in Docker.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

docker compose up -d db        # start PostgreSQL
python src/ingest.py           # download demand and weather data
python src/validate.py         # check data quality
python src/preprocess.py       # clean, align and join
python src/features.py         # build model features
python src/train.py            # train and compare with the baseline
python src/quantiles.py        # train and calibrate the prediction intervals
python src/price_features.py   # build price features
python src/train_price.py      # select features, train the price model, compare with baselines
python src/spikes.py           # train the spike classifier and choose its cut off
python src/battery_backtest.py          # backtest battery trading strategies on the test year
```

On macOS, LightGBM needs OpenMP first: `brew install libomp`

**2. Serve forecasts**, with the API and database together in Docker Compose:

```bash
docker compose up --build
```

or locally, with the database already running:

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive API documentation.

**3. Run the tests**

```bash
pytest
ruff check .
```

Every push runs CI on GitHub Actions; Ruff linting, a PostgreSQL database started with Docker Compose, the test suite with coverage, and the API container started with Docker Compose and checked by requesting a forecast. In CI, the tests run against a small model and features table built from synthetic data, since the real data and models aren't stored in the repository. The script that creates them refuses to run outside CI, so it can't overwrite real models. Dependabot opens weekly pull requests for dependency updates.

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` if the service is running |
| `GET /predict?day=YYYY-MM-DD` | Half-hourly forecast, 80% preditcion interval and actual demand for that day, with the day's MAE and MAPE |

Example response (trimmed):

```json
{
  "date": "2026-08-26",
  "in_training_data": false,
  "mae": 656,
  "mape": 2.67,
  "interval": "80%",
  "forecast": [
    {"timestamp_utc": "2026-08-25T23:00:00", "low": 20175, "forecast": 21673, "high": 22963, "actual": 22284}
  ]
}
```

`in_training_data` is `true` when the requested day was part of the training set, where results look better than the model's real performance. Invalid dates return 422, and dates outside the data return 404.

A single day can also be forecast from the command line: `python src/predict.py 2026-08-26`

---

## How it works

### Pipeline

| Stage | File | Purpose |
|---|---|---|
| Ingest | `src/ingest.py` | Downloads demand from Elexon in weekly chunks and weather from Open Meteo |
| Validate | `src/validate.py` | Checks for missing periods, duplicates, nulls, out of range values and date coverage |
| Explore | `notebooks/exploration.ipynb` | Daily, weekly and seasonal patterns, temperature relationship, autocorrelation, holidays |
| Preprocess | `src/preprocess.py` | Converts to UTC, fills gaps, resamples weather to half hourly, joins the sources |
| Features | `src/features.py` | Builds calendar, weather and lag features |
| Train | `src/train.py` | Trains LightGBM with a time based split and compares it with the baseline |
| Quantiles | `src/quantiles.py` | Trains q10/q50/q90 models and calibrates the prediction intervals |
| Retrain | `src/retrain.py` | Checks a newly trained model against the baseline on the latest 8 weeks, then retrains the demand and interval models on all data and recalibrates the intervals |
| Price features | `src/price_features.py` | Adds wind, solar, net demand and price lag features |
| Price model | `src/train_price.py` | Selects features on a validation year, trains LightGBM and compares it with two baselines |
| Spikes | `src/spikes.py` | Labels price spikes, selects features and a probability cut-off on a validation year, and compares the classifier with persistence |
| Backtest | `src/battery_backtest.py` | Plans each day's battery schedule from a price forecast with linear programming and scores it on real prices over the test year |
| Evaluate | `src/evaluate.py` | MAE, RMSE, MAPE and quantile loss |
| Predict | `src/predict.py` | Loads the saved models and forecasts a chosen day with interval |
| Serve | `api/main.py` | FastAPI service, packaged with the `Dockerfile` and run alongside PostgreSQL with `docker-compose.yml` |
| Test | `tests/` | Unit tests for metrics, features, the spike definition and the battery, plus API tests |
| CI | `.github/workflows/tests.yml` | Lints, runs the tests with coverage, and builds and starts the Docker image on every push |
| Scheduled retraining | `.github/workflows/retrain.yml` | Every Monday, downloads the latest data, rebuilds the features and runs the retraining, saving the new models as downloadable |

### Data

- **Demand:** initial national demand outturn from the Elexon BMRS API, half hourly, January 2023 to September 2026 (64,316 settlement periods).
- **Weather:** hourly temperature, wind speed and shortwave radiation for London from the Open Meteo historical archive, interpolated to half hourly.
- **Prices:** Market Index Price (APX/EPEX) from the Elexon BMRS API. Six missing half hours and 34 half hours with zero traded volume were filled by interpolation.
- **Wind and solar generation:** actual onshore wind, offshore wind and solar generation from the Elexon BMRS API. 4,383 republished duplicate rows were removed (keeping the latest version), 863 missing half-hours (longest gap 7 hours) were filled by interpolation, and 15 small negative values were set to zero.
- **Storage:** raw downloads are kept as CSV files, untouched. Everything from cleaning onwards is stored in PostgreSQL tables: `dataset`, `market`, `features` and `price_features`.

Validation found two days each missing one settlement period (2023-07-17 period 45 and 2023-12-29 period 8), which were filled by time interpolation.

### Features

| Feature | Description |
|---|---|
| `period` | Settlement period (1–48), in UK local time |
| `dayofweek`, `dayofyear` | Calendar position |
| `is_holiday` | England and Wales bank holidays |
| `is_christmas` | 24 December to 1 January |
| `temperature_2m` | Air temperature (°C) |
| `hdd` | Heating degree days: max(0, 15.5 − temperature) |
| `shortwave_radiation` | Sunlight reaching the ground (W/m²), a proxy for rooftop solar output |
| `demand_lag_recent` | Demand at the same time yesterday, or two days ago (see below) |
| `demand_lag_336` | Demand at the same time last week |
| `demand_roll_96` | 24 hour average demand ending two days earlier |

Wind speed was dropped because it had almost no correlation with demand (0.03).

### Price features

The price model uses the demand, weather and calendar features (without `dayofyear`) plus:

| Feature | Description |
|---|---|
| `wind_onshore`, `wind_offshore`, `wind_total` | Wind generation (MW) |
| `solar_generation` | Solar generation (MW) |
| `net_demand` | Demand minus wind and solar: what's left to be met mostly by gas, which usually sets the price |
| `price_lag_recent` | Price at the same time yesterday, or two days ago (same rule as demand) |
| `price_lag_336` | Price at the same time last week |
| `price_roll_7d` | 7 day average price ending two days earlier, which tells the model the current price level |

### Design decisions

**A realistic day-ahead forecast.** The forecast for a given day is assumed to be made around 9am the day before, in line with GB day ahead power auctions, which run in the morning. At that point, yesterday's demand is only known up to about 8am. So `demand_lag_recent` uses lag 48 (yesterday) for periods before 8am local time and lag 96 (two days ago) from 8am onwards. Using lag 48 everywhere would quietly use data a real forecaster wouldn't have yet. A unit test enforces this rule, and the rolling average is shifted by 96 periods for the same reason.

**Time-based split.** Training data ends on 31 August 2025 and testing starts on 1 September 2025, so the model is never trained on data from after the period it's tested on.

**Conformal calibration of the intervals.** Quantile models learn their spread from training data they fit closely, so their intervals come out too narrow on new data. To correct this, the quantile models were first trained on data up to August 2024 and evaluated on a held out calibration year (September 2024 to August 2025). The amount by which actual values fell outside their intervals gave a 758 MW adjustment, the widening needed for 80% of calibration values to fall inside. The final models were retrained on the full training period and the same adjustment applied.

**Timezones.** Timestamps are stored in UTC to avoid duplicate and missing hours at the clock changes. Calendar features use UK local time, because that's when people actually use electricity.

**Net demand for prices.** Prices are set by supply and demand together. Low demand alone doesn't explain negative prices (overnight demand is low every night, but prices go negative on only a small share of nights), whereas low demand combined with high wind or solar does. Net demand captures this directly.

**Separate market data.** Prices and generation are stored in their own processed file, so adding them didn't change the demand pipeline or its results.

**Spikes relative to the recent level.** Price levels drift by a factor of two between years, so a fixed threshold such as "above £150/MWh" would label most of early 2023 as spikes and almost nothing in 2024, and the classifier would simply learn which periods were expensive. Defining a spike as a jump above the recent 7 day average captures sudden, unusual prices whatever the current level.

**PostgreSQL for processed data.** Raw downloads stay as CSV files so everything can be rebuilt without calling the APIs again, but every processed table lives in one place, the database, so training, backtesting and the API always read the same data. The connection comes from a `DATABASE_URL` environment variable, so the same code runs on a laptop, inside Docker Compose and in CI.

**Scheduled retraining with a quality gate.** Demand patterns drift as solar capacity grows and behaviour changes, so the models are retrained weekly on the latest data. Before anything is saved, a model trained without the most recent 8 weeks must beat the same time last week baseline on them, otherwise the run fails and the previous models are kept. The interval calibration is recomputed on the same recent weeks. The READMEs results use data up to 1 September 2026 so they stay reproducible, only the retraining workflow fetches newer data.

---

## Limitations

- The test period uses actual weather, not weather forecasts, so live performance would be somewhat worse.
- Weather comes from London only, while national demand depends on weather across Great Britain, especially for solar.
- Holidays follow the England and Wales calendar; Scotland and Northern Ireland differ.
- The API replays past days from stored features. Live forecasting, which needs the latest demand data and weather forecasts, is planned as part of scheduled retraining.
- The feature data is stored inside the Docker image; a database is planned.
- Interval coverage holds over the test year as a whole, not on every day, so easy days are covered more often and unusual days (such as Christmas) less often. The choice to apply the calibration to retrained models was checked once against the test set.
- The price model has no gas price input, although gas usually sets GB power prices. The 7-day average price captures its effect only indirectly.
- Half hourly prices are noisy, and extreme spikes and negative prices remain hard to predict.
- A spike probability cut off chosen on one year transfers imperfectly to a more volatile year, in practice it would need recalibrating regularly.

## Future Roadmap

- MLflow experiment tracking
- Monitoring dashboard
- Cloud deployment

---

## Data sources and licences

- Contains BMRS data © Elexon Limited copyright and database right 2026. [Licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)
- Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
