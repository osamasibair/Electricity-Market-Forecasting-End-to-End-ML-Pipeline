# UK Electricity Demand Forecasting Pipeline

An end to end machine learning pipeline that forecasts Great Britain's half hourly electricity demand one day ahead, from raw public data to a tested, containerised API.

**LightGBM day-ahead model: 5.05% MAPE, about 40% lower error than a same-time-last-week baseline (8.40%).**

**Stack:** Python, pandas, LightGBM, FastAPI, Docker, pytest

---

## Results

Test period: 1 September 2025 to 1 September 2026 (17,566 half-hours never seen in training).

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive baseline (same time last week) | 2132 MW | 2878 MW | 8.40% |
| **LightGBM** | **1247 MW** | **1628 MW** | **5.05%** |

| Day | MAPE |
|---|---|
| Typical weekday (26 August 2026) | 2.62% |
| Christmas Day 2025 | 13.52% |

### Error analysis

- **Solar:** errors peaked in the middle of the day, at 2.0–2.4 GW around midday compared with about 1 GW overnight. Rooftop solar isn't measured in this demand data, so sunny days reduce the demand the grid sees. Adding shortwave radiation as a feature reduced MAPE from 5.66% to 5.04%.
- **Christmas:** Christmas Day had a MAPE of 14.8%, with only two Christmases in the training data. A Christmas-period flag reduced this to 13.5%, but the model still overestimates demand.
- **Peaks:** the model gets the shape of each day right but underestimates the extremes, forecasting too low at the morning ramp and the evening peak.

---

## Quick start

The data and trained model aren't stored in the repository, so the pipeline has to be run once to create them.

**1. Set up and run the pipeline**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/ingest.py        # download demand and weather data
python src/validate.py      # check data quality
python src/preprocess.py    # clean, align and join
python src/features.py      # build model features
python src/train.py         # train and compare with the baseline
```

On macOS, LightGBM needs OpenMP first: `brew install libomp`

**2. Serve forecasts**, either locally:

```bash
uvicorn api.main:app --reload
```

or with Docker:

```bash
docker build -t electricity-forecast .
docker run -p 8000:8000 electricity-forecast
```

Then open `http://127.0.0.1:8000/docs` for interactive API documentation.

**3. Run the tests**

```bash
pytest
```

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` if the service is running |
| `GET /predict?day=YYYY-MM-DD` | Half-hourly forecast and actual demand for that day, with the day's MAE and MAPE |

Example response (trimmed):

```json
{
  "date": "2026-08-26",
  "in_training_data": false,
  "mae": 656,
  "mape": 2.62,
  "forecast": [
    {"timestamp_utc": "2026-08-25T23:00:00", "forecast": 21673, "actual": 22284}
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
| Evaluate | `src/evaluate.py` | MAE, RMSE and MAPE |
| Predict | `src/predict.py` | Loads the saved model and forecasts a chosen day |
| Serve | `api/main.py` | FastAPI service, packaged with the `Dockerfile` |
| Test | `tests/` | Unit tests for metrics and features, plus API tests |

### Data

- **Demand:** initial national demand outturn from the Elexon BMRS API, half hourly, January 2023 to September 2026 (64,316 settlement periods).
- **Weather:** hourly temperature, wind speed and shortwave radiation for London from the Open Meteo historical archive, interpolated to half hourly.

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

### Design decisions

**A realistic day-ahead forecast.** The forecast for a given day is assumed to be made around 9am the day before, in line with GB day ahead power auctions, which run in the morning. At that point, yesterday's demand is only known up to about 8am. So `demand_lag_recent` uses lag 48 (yesterday) for periods before 8am local time and lag 96 (two days ago) from 8am onwards. Using lag 48 everywhere would quietly use data a real forecaster wouldn't have yet. A unit test enforces this rule, and the rolling average is shifted by 96 periods for the same reason.

**Time-based split.** Training data ends on 31 August 2025 and testing starts on 1 September 2025, so the model is never trained on data from after the period it's tested on.

**Timezones.** Timestamps are stored in UTC to avoid duplicate and missing hours at the clock changes. Calendar features use UK local time, because that's when people actually use electricity.

---

## Limitations

- The test period uses actual weather, not weather forecasts, so live performance would be somewhat worse.
- Weather comes from London only, while national demand depends on weather across Great Britain, especially for solar.
- Holidays follow the England and Wales calendar; Scotland and Northern Ireland differ.
- The API replays past days from stored features. Live forecasting, which needs the latest demand data and weather forecasts, is planned as part of scheduled retraining.
- The feature data is stored inside the Docker image; a database is planned.

## Future Roadmap

- Probabilistic (quantile) forecasts
- System price forecasting and a price spike classifier
- CI with GitHub Actions
- PostgreSQL storage
- Scheduled retraining
- MLflow experiment tracking
- Monitoring dashboard
- Cloud deployment

---

## Data sources and licences

- Contains BMRS data © Elexon Limited copyright and database right 2026. [Licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)
- Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
