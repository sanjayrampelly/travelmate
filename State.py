from typing import TypedDict, Optional


class TravelState(TypedDict, total=False):
    """Shared state that flows through every agent in the pipeline.
    Each node reads what it needs and writes its own piece back in —
    this is the 'Shared State / Cross-Agent Context' box in the diagram.
    """

    # ---- User input ----
    origin: str
    destination: str
    destination_airport: str  # IATA code, e.g. 'GOI' — flight tool needs this; weather/hotel use the city name
    departure_date: str  # YYYY-MM-DD
    return_date: str     # YYYY-MM-DD
    travelers: int

    # ---- Filled in by each agent as the pipeline runs ----
    flight_results: Optional[dict]
    hotel_results: Optional[dict]
    weather_results: Optional[dict]
    itinerary: Optional[str]
    final_response: Optional[str]

    # ---- Bookkeeping ----
    errors: list[str]