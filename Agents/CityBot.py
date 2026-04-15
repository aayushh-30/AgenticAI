import os
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from rich import print

# Load environment variables
load_dotenv()

# -------------------- TOOLS --------------------

@tool
def getWeather(city: str) -> str:
    """
    Fetch current weather of a given city.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "Missing OpenWeather API key."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"{city}: {temp}°C, {desc}"
        else:
            return f"Error: {data.get('message', 'City not found')}"
    except Exception as e:
        return f"Weather API error: {str(e)}"


@tool
def getNews(city: str) -> str:
    """
    Fetch latest news about a given city.
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.invoke(f"Latest breaking news in {city}")
        return results[:1000]  # limit output
    except Exception as e:
        return f"News fetch error: {str(e)}"


# -------------------- TOOL REGISTRY --------------------

availableTools = {
    "getWeather": getWeather,
    "getNews": getNews
}

# -------------------- MODEL --------------------

model = init_chat_model("groq:openai/gpt-oss-120b")
model_with_tools = model.bind_tools([getWeather, getNews])

# -------------------- CHAT LOOP --------------------

messages = []

print("[bold cyan]AI Agent Started! Type 'exit' to quit.[/bold cyan]\n")

while True:
    userInput = input("User : ")

    if userInput.lower() == "exit":
        print("[bold red]Exiting...[/bold red]")
        break

    messages.append(HumanMessage(content=userInput))

    # First model response
    model_res = model_with_tools.invoke(messages)

    # -------------------- TOOL HANDLING --------------------
    if model_res.tool_calls:
        for tool_call in model_res.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            confirmation = input(
                f"Agent wants to use [bold yellow]{tool_name}[/bold yellow] with {tool_args}. Confirm (Y/N): "
            )

            if confirmation.lower() == 'n':
                print("[red]Tool access denied.[/red]")
                continue

            # Execute tool
            try:
                tool_func = availableTools[tool_name]
                tool_output = tool_func.invoke(tool_args)
            except Exception as e:
                tool_output = f"Tool execution error: {str(e)}"

            # Add assistant tool call + tool response
            messages.append(model_res)
            messages.append(
                ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                )
            )

        # Final response after tool execution
        final_response = model_with_tools.invoke(messages)
        messages.append(final_response)

        print(f"[bold green]AI :[/bold green] {final_response.content}")

    # -------------------- NORMAL RESPONSE --------------------
    else:
        messages.append(model_res)
        print(f"[bold green]AI :[/bold green] {model_res.content}")


print(messages)