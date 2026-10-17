from types import SimpleNamespace

import numpy as np
import pandas as pd

from briefing import facts_text, forecast_facts, grounded, write_briefing


class FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.messages = self

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


def make_forecast():
    index = pd.date_range("2026-10-17 23:00", periods=48, freq="30min")
    level = 30000 + 5000 * np.sin(np.arange(48) / 48 * 2 * np.pi)
    return pd.DataFrame({"forecast": level, "low": level - 1000, "high": level + 1000}, index=index)


def make_daily():
    return pd.DataFrame(
        {"mape_7d": [8.16], "baseline_mape_7d": [8.82], "coverage_30d": [67.8], "worse_than_baseline": [0], "low_coverage": [1]},
        index=pd.to_datetime(["2026-10-17"]),
    )


def test_facts_cover_the_forecast_accuracy_and_alerts():
    facts = forecast_facts(make_forecast(), make_daily(), extra=350)
    assert facts["forecast date"] == "18 Oct 2026"
    assert facts["forecast peak"] == "35000 MW at 06:00"
    assert facts["interval widening from recent live errors"] == "350 MW"
    assert facts["7-day MAPE"] == "8.16%"
    assert facts["alerts"] == "30-day interval coverage is below 70%"


def test_grounded_accepts_given_and_rounded_numbers():
    facts = "7-day MAPE: 8.16%\nforecast peak: 35000 MW at 06:00"
    assert grounded("Demand should peak at 35000 MW at 06:00, and the 7-day MAPE is 8.16%.", facts)
    assert grounded("The 7-day MAPE is about 8.2%.", facts)


def test_grounded_rejects_a_made_up_number():
    facts = "7-day MAPE: 8.16%\nforecast peak: 35000 MW at 06:00"
    assert not grounded("Demand should peak at 36500 MW.", facts)
    assert not grounded("Demand peaks at 35 GW.", facts)


def test_write_briefing_keeps_a_grounded_reply():
    client = FakeClient("Coverage alert: 30-day interval coverage is 67.8%, below the 80% target. Demand should peak at 35000 MW at 06:00.")
    briefing = write_briefing(make_forecast(), make_daily(), 350, client)
    assert briefing["briefing"].iloc[0].startswith("Coverage alert")
    assert "8.16%" in client.requests[0]["messages"][0]["content"]


def test_write_briefing_drops_a_reply_with_invented_numbers():
    client = FakeClient("Demand should peak at 41000 MW tomorrow.")
    assert write_briefing(make_forecast(), make_daily(), 350, client) is None


def test_write_briefing_skips_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert write_briefing(make_forecast(), make_daily(), 350) is None


def test_facts_text_lists_one_fact_per_line():
    assert facts_text({"a": "1", "b": "2"}) == "a: 1\nb: 2"