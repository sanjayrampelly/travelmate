import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("weather-server")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class DailyForecast(BaseModel):
    date: str
    max_temp_c: float
    min_temp_c: float
    precipitation_mm: float
    weather_summary: str


class ForecastResult(BaseModel):
    success: bool
    location: str | None = None
    forecast: list[DailyForecast] = Field(default_factory=list)
    error: str | None = None


class ForecastInput(BaseModel):
    city: str = Field(description="City name, e.g. 'Goa' or 'Paris'", min_length=1, max_length=100)
    days: int = Field(default=5, description="Number of forecast days (1-7)", ge=1, le=7)


# rough WMO weather-code -> summary mapping, covers the common cases
WMO_SUMMARY = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 51: "Light drizzle", 61: "Light rain", 63: "Moderate rain",
    65: "Heavy rain", 71: "Light snow", 80: "Rain showers", 95: "Thunderstorm",
}


@mcp.tool()
async def get_weather_forecast(input: ForecastInput) -> ForecastResult:
    """Get a multi-day weather forecast for a city, using Open-Meteo.

    Use this when the user wants to know the weather/conditions expected at
    their travel destination. Geocodes the city name first, then fetches
    the forecast. Returns success=False with an explanation if the city
    can't be found or the API call fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            geo_resp = await client.get(
                GEOCODE_URL, params={"name": input.city, "count": 1}, timeout=10.0
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return ForecastResult(success=False, error=f"Could not find location '{input.city}'")

            place = geo_data["results"][0]
            lat, lon = place["latitude"], place["longitude"]
            location_label = f"{place['name']}, {place.get('country', '')}"

            forecast_resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "forecast_days": input.days,
                    "timezone": "auto",
                },
                timeout=10.0,
            )
            forecast_resp.raise_for_status()
            f_data = forecast_resp.json()["daily"]
    except httpx.HTTPStatusError as exc:
        return ForecastResult(success=False, error=f"Open-Meteo returned {exc.response.status_code}")
    except httpx.RequestError as exc:
        return ForecastResult(success=False, error=f"Network error: {exc}")
    except (KeyError, IndexError) as exc:
        return ForecastResult(success=False, error=f"Unexpected response shape: {exc}")

    days_out = [
        DailyForecast(
            date=f_data["time"][i],
            max_temp_c=f_data["temperature_2m_max"][i],
            min_temp_c=f_data["temperature_2m_min"][i],
            precipitation_mm=f_data["precipitation_sum"][i],
            weather_summary=WMO_SUMMARY.get(f_data["weathercode"][i], "Unknown"),
        )
        for i in range(len(f_data["time"]))
    ]

    return ForecastResult(success=True, location=location_label, forecast=days_out)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")