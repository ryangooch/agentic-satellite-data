#!/usr/bin/env python3
"""MCP server for weather data via Open-Meteo API.

Provides historical weather and forecast data for any lat/lon.
Designed for the agentic satellite analysis demo — the agent uses this
to correlate crop stress with weather events (drought, heat, frost).

Open-Meteo is free, no API key required.

Run standalone:  uv run python mcp_servers/weather.py
Configure in Claude Code settings as an MCP server (see README).
"""
import json
from datetime import datetime, timedelta

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "weather",
    instructions=(
        "Weather data server for agricultural analysis. "
        "Use get_historical_weather to check past conditions (precipitation, temperature, soil moisture) "
        "that may explain crop stress. Use get_forecast to assess upcoming weather risks. "
        "Use get_growing_season_summary for a seasonal overview of a field's weather."
    ),
)

OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"


def _format_weather_table(dates: list, data: dict, variables: list[str]) -> str:
    """Format weather data as a readable text table."""
    header = f"{'Date':<12}" + "".join(f"{v:>14}" for v in variables)
    lines = [header, "-" * len(header)]
    for i, date in enumerate(dates):
        row = f"{date:<12}"
        for var in variables:
            key = var.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            # Try to find the matching key in data
            val = None
            for data_key in data:
                if data_key == "time":
                    continue
                if val is None:
                    val = data[data_key][i] if i < len(data.get(data_key, [])) else None
                    if val is not None:
                        break
            # Actually let's just iterate properly
            row_vals = []
            for data_key in data:
                if data_key == "time":
                    continue
                if i < len(data[data_key]):
                    row_vals.append(data[data_key][i])
            row = f"{date:<12}" + "".join(f"{v:>14.1f}" if v is not None else f"{'N/A':>14}" for v in row_vals)
            break
        lines.append(row)
    return "\n".join(lines)


@mcp.tool()
async def get_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> str:
    """Get historical daily weather data for a location.

    Returns temperature (min/max/mean), precipitation, soil moisture, and
    evapotranspiration. Useful for understanding what weather conditions
    led to observed crop stress.

    Args:
        latitude: Latitude in decimal degrees (e.g. 36.944)
        longitude: Longitude in decimal degrees (e.g. -120.108)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
            "soil_moisture_0_to_7cm_mean",
        ]),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_HISTORICAL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])

    if not dates:
        return f"No historical data available for ({latitude}, {longitude}) from {start_date} to {end_date}."

    # Compute summary stats
    temps_max = [t for t in daily.get("temperature_2m_max", []) if t is not None]
    temps_min = [t for t in daily.get("temperature_2m_min", []) if t is not None]
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    et0 = [e for e in daily.get("et0_fao_evapotranspiration", []) if e is not None]
    soil = [s for s in daily.get("soil_moisture_0_to_7cm_mean", []) if s is not None]

    lines = [
        f"Historical Weather: ({latitude:.3f}, {longitude:.3f})",
        f"Period: {start_date} to {end_date} ({len(dates)} days)",
        "",
        "Summary:",
        f"  Temperature: {min(temps_min):.0f}–{max(temps_max):.0f}°F (avg {sum(t for t in daily.get('temperature_2m_mean', []) if t is not None)/max(len(temps_max),1):.0f}°F)",
        f"  Total precipitation: {sum(precip):.2f} in",
        f"  Days with rain (>0.01 in): {sum(1 for p in precip if p > 0.01)}",
        f"  Avg ET₀: {sum(et0)/max(len(et0),1):.2f} in/day",
    ]

    if soil:
        lines.append(f"  Soil moisture (0-7cm): {min(soil):.3f}–{max(soil):.3f} m³/m³ (avg {sum(soil)/len(soil):.3f})")

    # Water balance
    total_precip = sum(precip)
    total_et = sum(et0)
    balance = total_precip - total_et
    lines.extend([
        "",
        f"Water balance: {total_precip:.2f} in precip - {total_et:.2f} in ET₀ = {balance:+.2f} in",
    ])
    if balance < -2:
        lines.append("⚠️ Significant water deficit — crops likely need irrigation.")
    elif balance < 0:
        lines.append("Mild water deficit — may need supplemental irrigation.")
    else:
        lines.append("Positive water balance — natural rainfall may be sufficient.")

    # Heat stress days
    heat_days = sum(1 for t in temps_max if t > 100)
    if heat_days > 0:
        lines.append(f"🌡️ {heat_days} day(s) above 100°F — potential heat stress.")

    # Daily detail (last 14 days or all if shorter)
    show_dates = dates[-14:] if len(dates) > 14 else dates
    start_idx = len(dates) - len(show_dates)
    lines.extend(["", "Daily detail (most recent):"])
    lines.append(f"{'Date':<12} {'MaxF':>6} {'MinF':>6} {'Precip':>8} {'ET0':>8} {'Soil':>8}")
    lines.append("-" * 56)
    for i, d in enumerate(show_dates):
        idx = start_idx + i
        t_max = daily["temperature_2m_max"][idx]
        t_min = daily["temperature_2m_min"][idx]
        p = daily["precipitation_sum"][idx]
        e = daily.get("et0_fao_evapotranspiration", [None] * len(dates))[idx]
        s = daily.get("soil_moisture_0_to_7cm_mean", [None] * len(dates))[idx]
        lines.append(
            f"{d:<12} {t_max or 0:>5.0f}F {t_min or 0:>5.0f}F {p or 0:>7.2f}\" {e or 0:>7.2f}\" {s or 0:>7.3f}"
        )

    return "\n".join(lines)


@mcp.tool()
async def get_forecast(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> str:
    """Get weather forecast for the next N days.

    Returns temperature, precipitation probability, wind, and soil moisture forecast.
    Useful for planning field visits or predicting near-term crop stress.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        days: Number of forecast days (1-16, default 7)
    """
    days = min(max(days, 1), 16)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration",
        ]),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "wind_speed_unit": "mph",
        "timezone": "America/Los_Angeles",
        "forecast_days": days,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_FORECAST, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])

    if not dates:
        return f"No forecast available for ({latitude}, {longitude})."

    lines = [
        f"Weather Forecast: ({latitude:.3f}, {longitude:.3f})",
        f"Next {len(dates)} days:",
        "",
        f"{'Date':<12} {'MaxF':>6} {'MinF':>6} {'Precip':>8} {'P(rain)':>8} {'Wind':>8} {'ET0':>8}",
        "-" * 64,
    ]

    total_precip = 0
    heat_days = 0
    for i, d in enumerate(dates):
        t_max = daily["temperature_2m_max"][i] or 0
        t_min = daily["temperature_2m_min"][i] or 0
        p = daily["precipitation_sum"][i] or 0
        p_prob = daily["precipitation_probability_max"][i] or 0
        wind = daily["wind_speed_10m_max"][i] or 0
        et0 = daily.get("et0_fao_evapotranspiration", [0] * len(dates))[i] or 0
        total_precip += p
        if t_max > 100:
            heat_days += 1
        lines.append(
            f"{d:<12} {t_max:>5.0f}F {t_min:>5.0f}F {p:>7.2f}\" {p_prob:>6.0f}% {wind:>6.0f}mph {et0:>7.2f}\""
        )

    lines.extend(["", "Outlook:"])
    if total_precip < 0.1:
        lines.append("  No significant precipitation expected — irrigated fields should maintain schedules.")
    else:
        lines.append(f"  Total expected precipitation: {total_precip:.2f} in")
    if heat_days > 0:
        lines.append(f"  🌡️ {heat_days} day(s) above 100°F forecast — monitor for heat stress.")

    return "\n".join(lines)


@mcp.tool()
async def get_growing_season_summary(
    latitude: float,
    longitude: float,
    year: int,
) -> str:
    """Get a growing season weather summary (April-September) for a location.

    Provides accumulated precipitation, growing degree days, heat stress events,
    and overall water balance. Useful for contextualizing current crop conditions
    within the broader season.

    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        year: Year to summarize (e.g. 2024)
    """
    start = f"{year}-04-01"
    # Don't go past today
    today = datetime.now().strftime("%Y-%m-%d")
    end = min(f"{year}-09-30", today)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
        ]),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_HISTORICAL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])

    if not dates:
        return f"No data for {year} growing season at ({latitude}, {longitude})."

    temps_max = daily.get("temperature_2m_max", [])
    temps_min = daily.get("temperature_2m_min", [])
    temps_mean = daily.get("temperature_2m_mean", [])
    precip = daily.get("precipitation_sum", [])
    et0 = daily.get("et0_fao_evapotranspiration", [])

    # Growing degree days (base 50°F)
    gdd = sum(max((t or 50) - 50, 0) for t in temps_mean)
    total_precip = sum(p or 0 for p in precip)
    total_et = sum(e or 0 for e in et0)
    heat_days = sum(1 for t in temps_max if t and t > 100)
    frost_days = sum(1 for t in temps_min if t and t < 32)
    dry_spells = 0
    current_dry = 0
    max_dry = 0
    for p in precip:
        if (p or 0) < 0.01:
            current_dry += 1
            max_dry = max(max_dry, current_dry)
        else:
            if current_dry >= 14:
                dry_spells += 1
            current_dry = 0

    # Monthly breakdown
    monthly = {}
    for i, d in enumerate(dates):
        month = d[:7]
        if month not in monthly:
            monthly[month] = {"precip": 0, "et0": 0, "max_temps": [], "days": 0}
        monthly[month]["precip"] += precip[i] or 0
        monthly[month]["et0"] += et0[i] or 0
        monthly[month]["max_temps"].append(temps_max[i] or 0)
        monthly[month]["days"] += 1

    lines = [
        f"Growing Season Summary: ({latitude:.3f}, {longitude:.3f}) — {year}",
        f"Period: {dates[0]} to {dates[-1]} ({len(dates)} days)",
        "",
        f"  Growing Degree Days (base 50°F): {gdd:.0f}",
        f"  Total precipitation: {total_precip:.2f} in",
        f"  Total ET₀: {total_et:.2f} in",
        f"  Water balance: {total_precip - total_et:+.2f} in",
        f"  Heat stress days (>100°F): {heat_days}",
        f"  Frost days (<32°F): {frost_days}",
        f"  Longest dry spell: {max_dry} days",
        f"  Dry spells (>14 days): {dry_spells}",
        "",
        "Monthly breakdown:",
        f"{'Month':<10} {'Precip':>8} {'ET0':>8} {'AvgMax':>8} {'Days>100':>10}",
        "-" * 48,
    ]

    for month in sorted(monthly):
        m = monthly[month]
        avg_max = sum(m["max_temps"]) / max(len(m["max_temps"]), 1)
        hot_days = sum(1 for t in m["max_temps"] if t > 100)
        lines.append(
            f"{month:<10} {m['precip']:>7.2f}\" {m['et0']:>7.2f}\" {avg_max:>7.0f}F {hot_days:>10}"
        )

    return "\n".join(lines)


@mcp.tool()
async def get_cwsi_weather_data(
    latitude: float,
    longitude: float,
    date: str,
) -> str:
    """Get temperature and humidity data needed for CWSI (Crop Water Stress Index) calculation.

    Fetches hourly temperature and relative humidity for midday hours (10am-3pm)
    on the given date, computes VPD (vapor pressure deficit), and returns
    the values needed to calculate empirical CWSI.

    CWSI uses VPD to estimate crop water stress: higher VPD with low
    transpiration (visible as low NDVI) indicates water stress.

    Args:
        latitude: Latitude in decimal degrees (e.g. 36.944)
        longitude: Longitude in decimal degrees (e.g. -120.108)
        date: Date to analyze (YYYY-MM-DD)
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
        ]),
        "temperature_unit": "fahrenheit",
        "timezone": "America/Los_Angeles",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(OPEN_METEO_HISTORICAL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps_f = hourly.get("temperature_2m", [])
    rh_vals = hourly.get("relative_humidity_2m", [])

    if not times:
        return f"No hourly data available for ({latitude}, {longitude}) on {date}."

    # Filter to midday hours (10:00-15:00) when CWSI is most meaningful
    midday_temps_f = []
    midday_rh = []
    for i, t in enumerate(times):
        hour = int(t.split("T")[1].split(":")[0])
        if 10 <= hour <= 15 and temps_f[i] is not None and rh_vals[i] is not None:
            midday_temps_f.append(temps_f[i])
            midday_rh.append(rh_vals[i])

    if not midday_temps_f:
        return f"No midday data available for ({latitude}, {longitude}) on {date}."

    # Compute VPD for each midday hour
    # VPD = es - ea, where es = saturation vapor pressure, ea = actual vapor pressure
    # es(T) = 0.6108 * exp(17.27 * T_c / (T_c + 237.3))  [kPa, T in Celsius]
    vpd_vals = []
    for tf, rh in zip(midday_temps_f, midday_rh):
        tc = (tf - 32) * 5 / 9  # Convert F to C
        es = 0.6108 * (2.71828 ** (17.27 * tc / (tc + 237.3)))
        ea = es * (rh / 100)
        vpd_vals.append(es - ea)

    mean_temp_f = sum(midday_temps_f) / len(midday_temps_f)
    mean_rh = sum(midday_rh) / len(midday_rh)
    mean_vpd = sum(vpd_vals) / len(vpd_vals)
    max_vpd = max(vpd_vals)

    # Empirical CWSI estimates for common crops using VPD baselines
    # VPD_lower = non-stressed baseline, VPD_upper = fully-stressed limit
    # Based on Idso et al. (1981) and Jackson et al. (1981)
    crop_baselines = {
        "almond":  {"vpd_lower": 1.0, "vpd_upper": 4.5},
        "corn":    {"vpd_lower": 0.8, "vpd_upper": 3.5},
        "cotton":  {"vpd_lower": 1.2, "vpd_upper": 5.0},
        "grape":   {"vpd_lower": 0.9, "vpd_upper": 4.0},
        "tomato":  {"vpd_lower": 0.8, "vpd_upper": 3.8},
    }

    lines = [
        f"CWSI Weather Data: ({latitude:.3f}, {longitude:.3f}) on {date}",
        f"Midday hours (10am-3pm): {len(midday_temps_f)} observations",
        "",
        f"  Mean air temperature: {mean_temp_f:.1f}°F ({(mean_temp_f - 32) * 5 / 9:.1f}°C)",
        f"  Mean relative humidity: {mean_rh:.0f}%",
        f"  Mean VPD: {mean_vpd:.2f} kPa",
        f"  Max VPD: {max_vpd:.2f} kPa",
        "",
        "Empirical CWSI estimates by crop type:",
        f"  {'Crop':<10} {'CWSI':>6}  {'Status'}",
        f"  {'-'*35}",
    ]

    for crop, bl in crop_baselines.items():
        cwsi = max(0, min(1, (mean_vpd - bl["vpd_lower"]) / (bl["vpd_upper"] - bl["vpd_lower"])))
        if cwsi < 0.3:
            status = "low stress"
        elif cwsi < 0.6:
            status = "moderate stress"
        else:
            status = "HIGH STRESS"
        lines.append(f"  {crop:<10} {cwsi:>5.2f}  {status}")

    lines.extend([
        "",
        "Use these values with compute_cwsi tool:",
        f"  air_temp_f={mean_temp_f:.1f}, vpd_kpa={mean_vpd:.2f}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
