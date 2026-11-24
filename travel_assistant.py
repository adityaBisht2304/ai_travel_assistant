"""Gradio interface for the travel assistant."""

import gradio as gr

from travel_agent import chat_function


def main() -> None:
    # ChatInterface passes each prompt together with that session's prior messages.
    gr.ChatInterface(
        fn=chat_function,
        title="Travel Assistant",
        description=(
            "Get a trip plan with budget-matched hotels, food joints, local transport, "
            "and places to visit. Include your destination, trip length, traveller count, "
            "dates, and budget."
        ),
    ).launch()


if __name__ == "__main__":
    # Keep imports safe; launch the web app only when this file is executed directly.
    main()
