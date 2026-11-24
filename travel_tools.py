"""Travel research tools used by the travel assistant agent."""

from datetime import date, timedelta
from typing import Any

from strands import tool
from tavily import AsyncTavilyClient


def _extract_search_research(response: dict[str, Any]) -> dict[str, Any]:
    # Drop Tavily metadata but retain answer text and citation details.
    return {
        "answer": response.get("answer"),
        "sources": [
            {
                key: result[key]
                for key in ("title", "url", "content")
                if result.get(key)
            }
            for result in response.get("results", [])
            if isinstance(result, dict)
        ],
    }


def _build_trip_dates(start_date: date, days: int) -> list[dict[str, str]]:
    # Generate weekdays in Python so the assistant can use verified calendar dates.
    return [
        {
            "date": (start_date + timedelta(days=offset)).isoformat(),
            "weekday": (start_date + timedelta(days=offset)).strftime("%A"),
        }
        for offset in range(days)
    ]


def _resolve_start_date(
    day: int | None, month: int | None, year: int | None
) -> date:
    if day is None and month is None and year is None:
        return date.today()
    if month is None:
        raise ValueError("Provide a month; the day and year are optional.")
    if day is None:
        if year is not None:
            return date(year, month, 1)
        day = 1
    if year is not None:
        return date(year, month, day)

    # Resolve omitted years to the next valid occurrence, including leap days.
    today = date.today()
    candidate_year = today.year
    while True:
        try:
            candidate = date(candidate_year, month, day)
        except ValueError:
            if month != 2 or day != 29:
                raise
        else:
            if candidate >= today:
                return candidate
        candidate_year += 1


class TravelTools:
    def __init__(self) -> None:
        # Reuse the async search client across all tool calls.
        self.tavily_client = AsyncTavilyClient()

    @tool
    async def get_weather_forecast(
        self,
        city: str,
        days: int,
        startDateDay: int | None = None,
        startDateMonth: int | None = None,
        startDateYear: int | None = None,
    ) -> dict[str, Any]:
        """Get a weather forecast for the destination and trip dates.

        Args:
            city: Destination city name (e.g., "Goa", "Bangalore")
            days: Number of days in the trip
            startDateDay: Optional day of the month (1-31); defaults to the first
                of the month when a month is provided
            startDateMonth: Optional month (1-12)
            startDateYear: Optional year; if omitted, use the next future occurrence
        """
        start_date = _resolve_start_date(
            startDateDay, startDateMonth, startDateYear
        )

        if days < 1:
            raise ValueError("Days must be a positive number.")

        trip_dates = _build_trip_dates(start_date, days)
        end_date = trip_dates[-1]["date"]
        forecast = await self.tavily_client.search(
            query=(
                f"Weather forecast for {city} from {start_date.isoformat()} "
                f"through {end_date}. If a reliable forecast is unavailable for "
                "these dates, say so instead of inventing daily conditions."
            )
        )
        return {
            "start_date": start_date.isoformat(),
            "trip_dates": trip_dates,
            "forecast": forecast,
        }

    @tool
    async def suggest_packing_list(
        self, high_c: int, low_c: int, days: int, conditions: str
    ) -> list[str]:
        """Suggest what to pack based on temperatures, trip length and conditions.

        Args:
            high_c: Daytime high in Celsius
            low_c: Night-time low in Celsius
            days: Number of days in the trip
            conditions: Short description of expected weather
        """
        # Include trip essentials first, then tailor the list to forecast conditions.
        items = [f"{days + 1} sets of clothes", "toiletries", "phone charger"]

        # Add sun protection for hot daytime conditions.
        if high_c >= 30:
            items += [
                "light cotton clothing",
                "sunscreen",
                "sunglasses",
                "reusable water bottle",
            ]
        # Choose warm layers based on the expected overnight low.
        if low_c <= 15:
            items += ["warm jacket", "thermal layer"]
        elif low_c <= 22:
            items += ["light jacket for evenings"]
        # Pack rain gear when showers are expected.
        if "rain" in conditions.lower() or "shower" in conditions.lower():
            items += ["compact umbrella", "quick-dry footwear", "waterproof_jacket"]
        # Add winter gear when snow is expected.
        if "snow" in conditions.lower():
            items += ["gloves", "woollen cap", "waterproof boots"]

        return items

    @tool
    async def estimate_trip_cost(
        self, city: str, days: int, travellers: int, budget_inr: int
    ) -> dict[str, Any]:
        """Research budget-matched hotels, food joints, local transport, and attractions.

        Args:
            city: Destination city
            days: Number of days
            travellers: Number of people travelling
            budget_inr: Total trip budget in Indian rupees
        """
        if days < 1 or travellers < 1 or budget_inr < 1:
            raise ValueError("Days, travellers, and budget must all be positive.")

        # Estimate overnight stays and round up to one room per two travellers.
        nights = max(days - 1, 0)
        rooms_needed = (travellers + 1) // 2
        # Allocate 95% across categories and leave the remaining 5% as contingency.
        lodging_budget = budget_inr * 40 // 100
        food_budget = budget_inr * 30 // 100
        transport_budget = budget_inr * 15 // 100
        sightseeing_budget = budget_inr * 10 // 100
        # Turn category totals into per-night and per-person search limits.
        hotel_limit_per_night = lodging_budget // (max(nights, 1) * rooms_needed)
        food_limit_per_person_per_day = food_budget // (days * travellers)
        transport_limit_per_person_per_day = transport_budget // (days * travellers)

        # Search each spending category separately to keep the evidence relevant.
        stay_research = await self.tavily_client.search(
            query=(
                f"Find 3 budget hotels in {city}, India for two guests, with current "
                f"nightly room rates at or below INR {hotel_limit_per_night}. Include "
                "typical nightly prices and sources."
            ),
            include_answer="basic",
            max_results=5,
        )
        food_research = await self.tavily_client.search(
            query=(
                f"Find 3 affordable restaurants or local food joints in {city}, India "
                f"where a person can eat within INR {food_limit_per_person_per_day} "
                "per day. Include typical prices and sources."
            ),
            include_answer="basic",
            max_results=5,
        )
        local_travel_research = await self.tavily_client.search(
            query=(
                f"Average local transportation cost per person per day in {city}, India, "
                f"within INR {transport_limit_per_person_per_day}; include public "
                "transport, taxi or auto-rickshaw prices and sources."
            ),
            include_answer="basic",
            max_results=5,
        )
        places_research = await self.tavily_client.search(
            query=(
                f"Five best places to visit in {city}, India, with entry fees and "
                f"transport costs suitable for a group sightseeing budget of INR "
                f"{sightseeing_budget}. Include free or low-cost attractions and sources."
            ),
            include_answer="basic",
            max_results=5,
        )

        # Return both budget limits and source evidence for the agent's recommendations.
        return {
            "city": city,
            "days": days,
            "travellers": travellers,
            "budget_inr": budget_inr,
            "assumed_nights": nights,
            "rooms_needed": rooms_needed,
            "budget_limits_inr": {
                "lodging_total": lodging_budget,
                "hotel_room_per_night": hotel_limit_per_night,
                "food_per_person_per_day": food_limit_per_person_per_day,
                "local_transport_per_person_per_day": transport_limit_per_person_per_day,
                "sightseeing_group_total": sightseeing_budget,
                "contingency": budget_inr
                - lodging_budget
                - food_budget
                - transport_budget
                - sightseeing_budget,
            },
            "stay_research": _extract_search_research(stay_research),
            "food_research": _extract_search_research(food_research),
            "local_travel_research": _extract_search_research(local_travel_research),
            "places_research": _extract_search_research(places_research),
        }
