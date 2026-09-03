"""
WeatherGPT Core — Stage 1: CLI Weather Assistant
=================================================

This is the BRAIN of WeatherGPT. It connects a Large Language Model (LLM)
to REAL weather data from the Open-Meteo API (free, no API key needed).

╔══════════════════════════════════════════════════════════════════╗
║  CRITICAL DESIGN RULE (never break this):                       ║
║  The LLM NEVER invents or generates weather numbers.            ║
║  It ONLY narrates/explains data that came from the API.         ║
║  Accuracy = real API data. Intelligence = LLM language layer.   ║
╚══════════════════════════════════════════════════════════════════╝

How the 5-stage pipeline works:
  1. You type a weather question in natural language
  2. The LLM parses your intent and extracts the city name
  3. The get_weather() tool fetches REAL data from Open-Meteo
  4. The LLM reads the real data and writes a human-friendly answer
  5. You see the answer in your terminal

Supports: Anthropic Claude API  OR  OpenAI GPT API (auto-detects from env vars)

Usage:
  1. Set your API key:  set ANTHROPIC_API_KEY=sk-ant-...   (or OPENAI_API_KEY=sk-...)
  2. Run:               python weathergpt_core.py
  3. Type questions:    "Will it rain in Vizianagaram tomorrow?"
  4. Type 'quit' to exit

Author: WeatherGPT Team (SIH 2026)
"""

# ============================================================================
# IMPORTS — Libraries we need
# ============================================================================

import os          # For reading environment variables (API keys)
import sys         # For exiting the program cleanly
import json        # For converting data to/from JSON format
import requests    # For making HTTP requests to weather APIs

# python-dotenv lets us read API keys from a .env file (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # Ensures values in .env override any stale terminal env vars
except ImportError:
    pass  # dotenv not installed — that's fine, we'll read env vars directly


# ============================================================================
# CONFIGURATION — API Keys
# ============================================================================
# The script checks which API key you have set and uses that provider.
# You only need ONE of these — whichever LLM service you have access to.
#
# How to set (in your terminal, before running this script):
#   Windows CMD:        set ANTHROPIC_API_KEY=sk-ant-your-key-here
#   Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
#   Mac/Linux:          export ANTHROPIC_API_KEY=sk-ant-your-key-here
#
# Or create a file called ".env" in this folder with the line:
#   ANTHROPIC_API_KEY=sk-ant-your-key-here

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# ============================================================================
# WMO WEATHER CODE MAPPING
# ============================================================================
# Open-Meteo returns weather conditions as numeric codes (WMO international
# standard). This dictionary converts them to human-readable descriptions.
# For example: code 61 → "Slight rain", code 95 → "Thunderstorm"

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


# ============================================================================
# SYSTEM PROMPT — Instructions for the LLM
# ============================================================================
# This is the most important part of the entire project. It tells the LLM
# exactly how to behave. The "never invent numbers" rule lives here.

SYSTEM_PROMPT = """You are WeatherGPT, an intelligent weather assistant designed for users across India and worldwide.

━━━ YOUR #1 RULE (NEVER BREAK THIS) ━━━
• You must NEVER invent, estimate, guess, or hallucinate any weather data.
• This includes: temperatures, rainfall amounts, humidity, wind speed, forecasts, probabilities — ANY number related to weather.
• You may ONLY state weather values that were returned to you by the get_weather tool.
• If the user asks about weather and you haven't called get_weather yet, you MUST call it first before answering.
• If the tool returns an error, tell the user honestly. NEVER make up data to fill a gap.
• If the user asks about a date beyond the 3-day forecast range, say you don't have data for that date.

━━━ YOUR ROLE ━━━
• Understand the user's weather question (they may ask in any language or style)
• Call the get_weather tool with the correct city name
• Read the REAL data returned by the tool
• Provide a helpful, natural-language answer using ONLY those real numbers
• You may add helpful context (e.g., "that's quite warm for September" or "you may want an umbrella") but every weather number must come directly from the tool result

━━━ WHAT YOU CAN DO ━━━
• Current weather for any city worldwide
• 3-day weather forecast (today + next 2 days)
• Rain predictions ("will it rain tomorrow in Chennai?")
• Farming/agriculture advice based on real weather data ("should I sow paddy this week?")
• Travel weather checks ("what should I pack for a trip to Manali?")
• General weather-based recommendations (always grounded in real data)

━━━ WHAT YOU CANNOT DO (be honest about these) ━━━
• Historical weather data (the tool only gives current + 3-day forecast)
• Severe weather alerts (coming in a future update)
• Forecasts beyond 3 days from today
• Climate change analysis or long-term trends

━━━ ANSWERING STYLE ━━━
• Be warm, conversational, and helpful
• Always mention the city name in your answer (in case geocoding resolved to a different place)
• Include specific numbers: temperature, humidity, rain chance, wind speed
• For farming questions: give practical crop-weather advice based on real data
• Keep answers concise but informative — this is a chat, not an essay
• If the user greets you or asks a non-weather question, respond politely and let them know you specialize in weather
• Mention the date/day when discussing forecasts so the user knows which day you mean
"""


# ============================================================================
# WEATHER TOOL — Fetches REAL data from Open-Meteo (free, no API key needed)
# ============================================================================
# This function is the ONLY source of weather numbers in the entire system.
# The LLM calls this tool, and then uses its output to form an answer.

def get_weather(city_name: str) -> str:
    """
    Fetches REAL weather data for a given city from the Open-Meteo API.

    This is a two-step process:
      Step 1: Convert city name → latitude/longitude (geocoding)
      Step 2: Fetch current weather + 3-day forecast using those coordinates

    Args:
        city_name: Name of the city (e.g., "Mumbai", "Vizianagaram", "Delhi")

    Returns:
        A JSON string containing structured weather data, or an error message.
        The LLM will read this JSON and convert it to a natural language answer.
    """

    # ---- Step 1: GEOCODING — Convert city name to GPS coordinates ----
    # We need latitude and longitude to query the weather API.
    # Open-Meteo provides a free geocoding endpoint for this.

    try:
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city_name,   # The city we're looking for
            "count": 1,          # Only need the top/best match
            "language": "en",    # Return place names in English
            "format": "json"
        }

        print(f"  🔍 Looking up coordinates for '{city_name}'...")
        geo_response = requests.get(geocoding_url, params=geo_params, timeout=10)
        geo_response.raise_for_status()  # Throw an error if HTTP request failed
        geo_data = geo_response.json()

        # Check if the city was found
        if "results" not in geo_data or len(geo_data["results"]) == 0:
            return json.dumps({
                "error": True,
                "message": f"Could not find a city named '{city_name}'. "
                           f"Please check the spelling or try a nearby major city."
            })

        # Extract the location details from the first (best) match
        location = geo_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_name = location.get("name", city_name)
        country = location.get("country", "Unknown")
        state = location.get("admin1", "")  # State/province name

        print(f"  📍 Found: {resolved_name}, {state}, {country} "
              f"({latitude:.2f}°N, {longitude:.2f}°E)")

    except requests.exceptions.Timeout:
        return json.dumps({
            "error": True,
            "message": "The geocoding service took too long to respond. Please try again."
        })
    except requests.exceptions.ConnectionError:
        return json.dumps({
            "error": True,
            "message": "Could not connect to the geocoding service. Check your internet connection."
        })
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "error": True,
            "message": f"Geocoding failed with error: {str(e)}"
        })

    # ---- Step 2: FETCH WEATHER — Get current conditions + 3-day forecast ----
    # Now that we have coordinates, we can query the weather forecast API.

    try:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,

            # Current weather variables we want
            "current": ",".join([
                "temperature_2m",           # Air temperature at 2 meters height (°C)
                "relative_humidity_2m",     # Humidity percentage
                "apparent_temperature",     # "Feels like" temperature (°C)
                "precipitation",            # Current precipitation in mm
                "weather_code",             # WMO weather condition code
                "wind_speed_10m",           # Wind speed at 10m height (km/h)
                "wind_direction_10m",       # Wind direction in degrees
                "cloud_cover",              # Cloud cover percentage
            ]),

            # Daily forecast variables (for the next 3 days)
            "daily": ",".join([
                "weather_code",                 # Condition for the day
                "temperature_2m_max",           # Day's high temperature (°C)
                "temperature_2m_min",           # Day's low temperature (°C)
                "precipitation_sum",            # Total rainfall for the day (mm)
                "precipitation_probability_max",# Highest rain chance for the day (%)
                "sunrise",                      # Sunrise time
                "sunset",                       # Sunset time
                "uv_index_max",                 # Maximum UV index
                "wind_speed_10m_max",           # Maximum wind speed (km/h)
            ]),

            "timezone": "auto",       # Auto-detect timezone from coordinates
            "forecast_days": 3,       # Today + next 2 days
        }

        print(f"  🌤️  Fetching weather data...")
        weather_response = requests.get(forecast_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

    except requests.exceptions.Timeout:
        return json.dumps({
            "error": True,
            "message": "The weather service took too long to respond. Please try again."
        })
    except requests.exceptions.ConnectionError:
        return json.dumps({
            "error": True,
            "message": "Could not connect to the weather service. Check your internet connection."
        })
    except requests.exceptions.RequestException as e:
        return json.dumps({
            "error": True,
            "message": f"Weather data fetch failed with error: {str(e)}"
        })

    # ---- Step 3: STRUCTURE THE RESPONSE ----
    # Organize the raw API data into a clean, readable format that the LLM
    # can easily understand and convert to natural language.

    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    # Decode the weather code into a human-readable condition string
    current_code = current.get("weather_code", -1)
    current_condition = WMO_WEATHER_CODES.get(
        current_code, f"Unknown condition (code {current_code})"
    )

    # Build the structured result
    result = {
        "source": "Open-Meteo API (real data, not AI-generated)",
        "location": {
            "city": resolved_name,
            "state": state,
            "country": country,
            "coordinates": f"{latitude:.4f}°N, {longitude:.4f}°E",
            "timezone": weather_data.get("timezone", "Unknown"),
        },
        "current_weather": {
            "temperature_celsius": current.get("temperature_2m"),
            "feels_like_celsius": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "cloud_cover_percent": current.get("cloud_cover"),
            "condition": current_condition,
            "observation_time": current.get("time", "Unknown"),
        },
        "three_day_forecast": [],
    }

    # Build the 3-day forecast array
    dates = daily.get("time", [])
    for i in range(len(dates)):
        # Safely get each daily value (handles missing data gracefully)
        def safe_get(key, index):
            """Helper to safely extract a value from the daily arrays."""
            arr = daily.get(key, [])
            return arr[index] if index < len(arr) else None

        day_code = safe_get("weather_code", i)
        day_condition = WMO_WEATHER_CODES.get(day_code, "Unknown") if day_code is not None else "Unknown"

        day_data = {
            "date": dates[i],
            "condition": day_condition,
            "max_temp_celsius": safe_get("temperature_2m_max", i),
            "min_temp_celsius": safe_get("temperature_2m_min", i),
            "precipitation_total_mm": safe_get("precipitation_sum", i),
            "rain_chance_percent": safe_get("precipitation_probability_max", i),
            "max_wind_speed_kmh": safe_get("wind_speed_10m_max", i),
            "uv_index_max": safe_get("uv_index_max", i),
            "sunrise": safe_get("sunrise", i),
            "sunset": safe_get("sunset", i),
        }
        result["three_day_forecast"].append(day_data)

    print(f"  ✅ Weather data retrieved successfully!")

    # Return as a JSON string — this is what the LLM will read and narrate
    return json.dumps(result, indent=2)


# ============================================================================
# TOOL DEFINITIONS — Tell the LLM what tools it can call
# ============================================================================
# These definitions describe the get_weather function to the LLM so it knows:
#   - The tool exists and what it does
#   - What arguments it accepts
#   - When to call it (based on user's question)

# Tool definition in ANTHROPIC format (Claude API)
ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Fetches real-time weather data and a 3-day forecast for any city "
            "worldwide. Returns current temperature, humidity, wind, conditions, "
            "and daily forecasts with rain probability. Call this tool whenever "
            "the user asks anything about weather, temperature, rain, forecast, "
            "or climate conditions for a specific location. The data comes from "
            "the Open-Meteo API and is 100% real — never fabricated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": (
                        "The name of the city to get weather for. "
                        "Examples: 'Mumbai', 'Vizianagaram', 'New Delhi', 'Chennai'. "
                        "Use the most common English spelling of the city name."
                    ),
                }
            },
            "required": ["city_name"],
        },
    }
]

# Tool definition in OPENAI format (GPT API)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Fetches real-time weather data and a 3-day forecast for any city "
                "worldwide. Returns current temperature, humidity, wind, conditions, "
                "and daily forecasts with rain probability. Call this tool whenever "
                "the user asks anything about weather, temperature, rain, forecast, "
                "or climate conditions for a specific location. The data comes from "
                "the Open-Meteo API and is 100% real — never fabricated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": (
                            "The name of the city to get weather for. "
                            "Examples: 'Mumbai', 'Vizianagaram', 'New Delhi', 'Chennai'. "
                            "Use the most common English spelling of the city name."
                        ),
                    }
                },
                "required": ["city_name"],
            },
        },
    }
]


# ============================================================================
# MAP OF AVAILABLE TOOLS — Used to execute tool calls from the LLM
# ============================================================================
# When the LLM decides to call a tool, we look up the function here and run it.

AVAILABLE_TOOLS = {
    "get_weather": get_weather,
}


# ============================================================================
# ANTHROPIC CLAUDE — Chat function with tool calling
# ============================================================================

def chat_with_anthropic(conversation_history: list) -> str:
    """
    Sends the conversation to Anthropic Claude and handles tool calling.

    Flow:
      1. Send user message + tool definitions to Claude
      2. If Claude wants to call a tool → execute it → send result back → get final answer
      3. If Claude responds directly (no tool needed) → return the text

    Args:
        conversation_history: List of message dicts [{"role": "user"/"assistant", "content": ...}]

    Returns:
        The assistant's final text response (after any tool calls are resolved)
    """
    # Import the Anthropic library (only when needed)
    try:
        import anthropic
    except ImportError:
        print("\n❌ ERROR: The 'anthropic' library is not installed.")
        print("   Run: pip install anthropic")
        sys.exit(1)

    # Create the API client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Send the conversation to Claude with our tool definitions
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",    # Using Claude Sonnet (fast + capable)
        max_tokens=1024,                     # Max length of response
        system=SYSTEM_PROMPT,                # Our "never invent numbers" instructions
        tools=ANTHROPIC_TOOLS,               # Tell Claude about our get_weather tool
        messages=conversation_history,       # The conversation so far
    )

    # --- Handle the response ---
    # Claude's response can contain multiple "content blocks":
    #   - TextBlock: Regular text response
    #   - ToolUseBlock: Claude wants to call one of our tools

    # Check if Claude wants to call a tool
    if response.stop_reason == "tool_use":
        # Claude decided it needs to call get_weather to answer the question

        # Add Claude's response (with tool request) to conversation history
        # We need to preserve the FULL response including both text and tool_use blocks
        conversation_history.append({
            "role": "assistant",
            "content": response.content,  # Contains ToolUseBlock(s)
        })

        # Process each tool call in the response
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name       # e.g., "get_weather"
                tool_input = block.input     # e.g., {"city_name": "Vizianagaram"}
                tool_use_id = block.id       # Unique ID to match result to request

                print(f"\n  🔧 LLM is calling tool: {tool_name}({tool_input})")

                # Execute the actual tool function
                if tool_name in AVAILABLE_TOOLS:
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_input)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                # Collect the result
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result,
                })

        # Send the tool result(s) back to Claude so it can write the final answer
        conversation_history.append({
            "role": "user",
            "content": tool_results,
        })

        # Get Claude's final response (now with real weather data)
        final_response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOLS,
            messages=conversation_history,
        )

        # Extract the text from the final response
        assistant_text = ""
        for block in final_response.content:
            if hasattr(block, "text"):
                assistant_text += block.text

        # Add the final answer to conversation history (for multi-turn memory)
        conversation_history.append({
            "role": "assistant",
            "content": final_response.content,
        })

        return assistant_text

    else:
        # Claude responded directly without needing a tool (e.g., for greetings)
        assistant_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                assistant_text += block.text

        # Add to conversation history
        conversation_history.append({
            "role": "assistant",
            "content": response.content,
        })

        return assistant_text


# ============================================================================
# OPENAI GPT — Chat function with tool calling
# ============================================================================

def chat_with_openai(conversation_history: list) -> str:
    """
    Sends the conversation to OpenAI GPT and handles tool calling.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("\n❌ ERROR: The 'openai' library is not installed.")
        print("   Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)
    messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=2048, tools=OPENAI_TOOLS, messages=messages_with_system
    )
    message = response.choices[0].message

    if message.tool_calls:
        conversation_history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"\n  🔧 LLM is calling tool: {tool_name}({tool_args})")

            tool_result = AVAILABLE_TOOLS[tool_name](**tool_args) \
                          if tool_name in AVAILABLE_TOOLS \
                          else json.dumps({"error": f"Unknown tool: {tool_name}"})

            conversation_history.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        final_response = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=2048, messages=messages_with_system
        )
        assistant_text = final_response.choices[0].message.content or ""
        conversation_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    else:
        assistant_text = message.content or ""
        conversation_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text


def chat_with_groq(conversation_history: list) -> str:
    """
    Sends the conversation to Groq Cloud (Free & High Speed) and handles tool calling.
    """
    import re
    try:
        from openai import OpenAI
    except ImportError:
        print("\n❌ ERROR: The 'openai' library is not installed.")
        print("   Run: pip install openai")
        sys.exit(1)

    key = GROQ_API_KEY or (ANTHROPIC_API_KEY if (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.startswith("gsk_")) else "")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)

    messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b", max_tokens=2048, tools=OPENAI_TOOLS, messages=messages_with_system
    )
    message = response.choices[0].message

    if message.tool_calls:
        conversation_history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"\n  🔧 LLM is calling tool: {tool_name}({tool_args})")

            tool_result = AVAILABLE_TOOLS[tool_name](**tool_args) \
                          if tool_name in AVAILABLE_TOOLS \
                          else json.dumps({"error": f"Unknown tool: {tool_name}"})

            conversation_history.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

        messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        final_response = client.chat.completions.create(
            model="qwen/qwen3.6-27b", max_tokens=2048, messages=messages_with_system
        )
        raw_text = final_response.choices[0].message.content or ""
        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        conversation_history.append({"role": "assistant", "content": clean_text})
        return clean_text
    else:
        raw_text = message.content or ""
        clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        conversation_history.append({"role": "assistant", "content": clean_text})
        return clean_text


# ============================================================================
# MAIN — The command-line chat loop
# ============================================================================

def main():
    """
    Main entry point. Detects which LLM provider is available,
    then starts an interactive chat loop in the terminal.
    """

    # ---- Detect which LLM provider to use ----
    if (GROQ_API_KEY and not GROQ_API_KEY.startswith("your-")) or (ANTHROPIC_API_KEY and ANTHROPIC_API_KEY.startswith("gsk_")):
        provider = "groq"
        provider_name = "Groq AI (Free & Fast)"
        chat_function = chat_with_groq
    elif ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("your-"):
        provider = "anthropic"
        provider_name = "Anthropic Claude"
        chat_function = chat_with_anthropic
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-"):
        provider = "openai"
        provider_name = "OpenAI GPT"
        chat_function = chat_with_openai
    else:
        # No API key found — show helpful instructions and exit
        print("=" * 65)
        print("  ❌ ERROR: No LLM API key found!")
        print("=" * 65)
        print()
        print("  WeatherGPT needs an LLM API key to understand your questions.")
        print("  You need ONE of the following:")
        print()
        print("  Option A — Anthropic Claude (recommended):")
        print("    1. Get a key at: https://console.anthropic.com/")
        print("    2. Set it:")
        print('       PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"')
        print('       CMD:         set ANTHROPIC_API_KEY=sk-ant-your-key-here')
        print()
        print("  Option B — OpenAI GPT:")
        print("    1. Get a key at: https://platform.openai.com/api-keys")
        print("    2. Set it:")
        print('       PowerShell:  $env:OPENAI_API_KEY="sk-your-key-here"')
        print('       CMD:         set OPENAI_API_KEY=sk-your-key-here')
        print()
        print("  Or create a .env file in this folder with:")
        print("    ANTHROPIC_API_KEY=sk-ant-your-key-here")
        print("=" * 65)
        sys.exit(1)

    # ---- Welcome banner ----
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║   🌤️  WeatherGPT — Your AI Weather Assistant               ║")
    print("║                                                            ║")
    print("║   Ask me about weather anywhere in the world!              ║")
    print("║   I use REAL data from Open-Meteo (never made-up numbers)  ║")
    print("║                                                            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║   LLM Provider: {provider_name:<43} ║")
    print("║   Weather Data: Open-Meteo API (free, real-time)           ║")
    print("║   Forecast Range: Current + 3 days                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                                                            ║")
    print("║   Try asking:                                              ║")
    print("║   • Will it rain in Vizianagaram tomorrow?                 ║")
    print("║   • What's the weather in Mumbai right now?                ║")
    print("║   • Should I carry an umbrella in Chennai today?           ║")
    print("║   • Is it good weather for sowing paddy in Guntur?        ║")
    print("║                                                            ║")
    print("║   Type 'quit' or 'exit' to stop.                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ---- Conversation history ----
    # This list stores all messages so the LLM remembers previous questions.
    # For example, if you ask "What's the weather in Delhi?" and then
    # "What about Mumbai?", the LLM knows you're comparing cities.
    conversation_history = []

    # ---- Main chat loop ----
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # User pressed Ctrl+C or Ctrl+D — exit gracefully
            print("\n\n👋 Goodbye! Stay weather-aware!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\n👋 Goodbye! Stay weather-aware!")
            break

        # Add user's message to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_input,
        })

        # Send to LLM and get response
        try:
            print()  # Blank line for readability
            response = chat_function(conversation_history)
            print(f"\n🌤️ WeatherGPT: {response}\n")

        except Exception as e:
            # Catch ANY error so the app never crashes during a demo
            error_msg = str(e)
            print(f"\n⚠️  Something went wrong: {error_msg}")

            # Give helpful hints based on common errors
            if "401" in error_msg or "invalid" in error_msg.lower() or "auth" in error_msg.lower():
                print("   → Your API key might be invalid. Double-check it.")
            elif "429" in error_msg or "rate" in error_msg.lower():
                print("   → Rate limit hit. Wait a moment and try again.")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                print("   → Network issue. Check your internet connection.")
            else:
                print("   → Try again, or type 'quit' to exit.")
            print()

            # Remove the failed message from history so it doesn't corrupt future calls
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()


# ============================================================================
# ENTRY POINT — This runs when you execute: python weathergpt_core.py
# ============================================================================

if __name__ == "__main__":
    main()
