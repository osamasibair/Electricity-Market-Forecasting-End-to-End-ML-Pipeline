"""Write a short briefing on the live forecasts with claude and reject it if it uses any number that isnt in the database."""
import os
import re

import pandas as pd

model_name = "claude-haiku-5-5"

instructions = (
    "You write a briefing of two or three sentences for the person who runs a uk electricity demand forecasting system."
    "Use only the facts you are given. Copy every number exactly as written with its unit and do not convert units,"
    "add numbers or guess at causes the facts dont mention. Write times in 24 hour form exactly as given."
    "If there are alerts start with them."
)


def forecast_facts(forecast, daily, extra):
    local = forecast.index.tz_localize("UTC").tz_convert("Europe/London")
    peak, low = forecast["forecast"].idxmax(), forecast["forecast"].idxmin()
    facts = {
        "forecast date": f"{local[0]:%d %b %Y}",
        "forecast peak": f"{forecast['forecast'].max():.0f} MW at {local[forecast.index.get_loc(peak)]:%H:%M}",
        "forecast low": f"{forecast['forecast'].min():.0f} MW at {local[forecast.index.get_loc(low)]:%H:%M}",
        "80% interval average width": f"{(forecast['high'] - forecast['low']).mean():.0f} MW",
        "interval widening from recent live errors": f"{extra:.0f} MW",
    }
    if len(daily):
        latest = daily.iloc[-1]
        facts["7-day MAPE"] = f"{latest['mape_7d']:.2f}%"
        facts["7-day baseline MAPE"] = f"{latest['baseline_mape_7d']:.2f}%"
        facts["30-day interval coverage"] = f"{latest['coverage_30d']:.1f}% (target 80%)"
        facts["alerts"] = "; ".join(alert_messages(latest)) or "none"
    return facts


def alert_messages(latest):
    alerts = []
    if latest["worse_than_baseline"]:
        alerts.append("the 7-day error is no better than the baseline")
    if latest["low_coverage"]:
        alerts.append("30-day interval coverage is below 70%")
    return alerts


def facts_text(facts):
    return "\n".join(f"{name}: {value}" for name, value in facts.items())


def numbers(text):
    return [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def ungrounded(reply, facts):
    allowed = [float(n) for n in numbers(facts)]
    missing = []
    for found in numbers(reply):
        decimals = len(found.split(".")[1]) if "." in found else 0
        if not any(round(fact, decimals) == float(found) for fact in allowed):
            missing.append(found)
    return missing


def grounded(reply, facts):
    return not ungrounded(reply, facts)


def ask_claude(client, facts):
    response = client.messages.create(
        model=model_name,
        max_tokens=2000,
        system=instructions,
        messages=[{"role": "user", "content": f"Facts:\n{facts}"}],
    )
    reply = "".join(block.text for block in response.content if block.type == "text").strip()
    if not reply:
        print(f"Claude returned no text (stop reason: {response.stop_reason})")
    return reply


def write_briefing(forecast, daily, extra, client=None):
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        import anthropic
        client = anthropic.Anthropic()
    facts = facts_text(forecast_facts(forecast, daily, extra))
    reply = ask_claude(client, facts)
    if not reply:
        return None
    missing = ungrounded(reply, facts)
    if missing:
        print(f"briefing rejected, these numbers aren't in the facts: {', '.join(missing)}\nreply: {reply}")
        return None
    return pd.DataFrame({"briefing": [reply], "facts": [facts]}, index=[forecast.index.min()])