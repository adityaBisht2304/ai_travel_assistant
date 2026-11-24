"""Travel-planning agent and the Gradio-compatible chat callback."""

import os
from typing import Any

from dotenv import load_dotenv
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.types.content import Message

from travel_tools import TravelTools

load_dotenv()
# Let users override the model without changing source code.
MODEL_ID = os.environ.get(
    "MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)

# Keep one tool set so its search client can be reused between turns.
travel_tools = TravelTools()

# Require source-backed suggestions and a consistent structure for each trip plan.
SYSTEM_PROMPT = (
    "You are a practical travel assistant. "
    "When asked about a trip: check the weather first, then suggest what to pack "
    "based on that weather, then estimate the cost. "
    "If the user does not provide travel dates, do not ask for them; use today's date "
    "for the weather search and tell the user which date you used. "
    "The weather tool returns trip_dates with authoritative ISO dates and weekdays. "
    "Use those exact dates and weekdays in the itinerary; never calculate or guess "
    "weekdays yourself. If weather research cannot verify a forecast for the requested "
    "dates, state that a reliable forecast is unavailable rather than presenting "
    "historical averages or invented daily conditions as a forecast. "
    "When a budget is provided, pass it in INR to the cost tool. Use its category "
    "limits and research to estimate trip costs, and recommend up to 3 hotels and "
    "3 food joints that have source evidence of prices within those limits. Include "
    "the listed prices and source URLs when available. Recommend places to visit from "
    "the attractions research, considering entry and local transport costs. If sources "
    "cannot verify a price or three suitable options, say so instead of inventing one. "
    "Assume days minus one hotel nights for a multi-day trip, with one room per two "
    "travellers; account for the required number of rooms in the lodging estimate. "
    "Multiply per-person food and transport costs by trip length and "
    "traveller count."
    "Provide the answer in following sections:-" 
    "1. Weather"
    "2. What to Pack"
    "3. Hotel Options"
    "4. Food Options"
    "5. Local Transport Options"
    "6. Places to Visit"
    "7. Itienary for days of Visit"
    "8. Analysis of Budget"
    "9. Recommendations"

    "Places to Visit will give a brief description about each place and time needed for that place"
    "Analysis of Budget section will cover the costing for each of the major sections above.  State budget-allocation assumptions and uncertainty."
    "Always say whether the trip fits the user's budget, and keep advice concise in Recommendations Section"
)


def _text_from_gradio_content(content: Any) -> str:
    # Keep plain text and text blocks; ignore non-text attachments in chat messages.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    return ""


def _build_conversation(message: str, history: list[dict[str, Any]]) -> list[Message]:
    conversation: list[Message] = []
    # Replay visible chat turns, skipping system/tool entries and empty content.
    for turn in history:
        role = turn.get("role")
        if role != "user" and role != "assistant":
            continue

        text = _text_from_gradio_content(turn.get("content"))
        if not text:
            continue

        # Strands represents message text as a list of content blocks.
        message_content = [{"text": text}]
        conversation_message = {"role": role, "content": message_content}
        conversation.append(conversation_message)

    # Gradio passes the new prompt separately from the prior history.
    current_message_content = [{"text": message}]
    current_message = {"role": "user", "content": current_message_content}
    conversation.append(current_message)
    return conversation


def chat_function(message: str, history: list[dict[str, Any]]) -> str:
    conversation = _build_conversation(message, history)
    # A fresh agent avoids mixing this session with another session's internal state.
    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID),
        messages=conversation,
        tools=[
            travel_tools.get_weather_forecast,
            travel_tools.suggest_packing_list,
            travel_tools.estimate_trip_cost,
        ],
        system_prompt=SYSTEM_PROMPT,
    )
    # Gradio expects text, while Strands returns an AgentResult object.
    return str(agent())
