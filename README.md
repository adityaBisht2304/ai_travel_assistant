# AI Travel Assistant

A Gradio chat app that uses a Strands agent, AWS Bedrock, and Tavily web search to help plan trips. It researches weather, packing suggestions, budget-matched hotels and food, local transport, and attractions.

## Requirements

- Python 3.10 or newer
- AWS credentials and a configured AWS region with access to the selected Bedrock model
- A Tavily API key

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```dotenv
TAVILY_API_KEY=your_tavily_api_key
MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

`MODEL_ID` is optional; the value shown is the default. Configure AWS credentials and the region through your normal AWS SDK or AWS CLI setup. Do not put AWS credentials in source control.

## Run

```powershell
python travel_assistant.py
```

Open the local URL printed by Gradio.

## Date handling

- If no travel date is supplied, weather research starts today.
- If only a month is supplied, the trip starts on the first day of its next occurrence. If the first day of that month has already passed this year, the next year's month is used.
- If a day and month are supplied without a year, the next occurrence of that date is used. If a day is omitted, the first day of the supplied month is used.
- A supplied year is honored. The month is required whenever any date part is given.

The app calculates each trip date and weekday in Python. Weather details come from web search and may not be available or reliable for dates outside a provider's forecast window; the assistant should report when it cannot verify a forecast.
