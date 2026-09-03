"""
WeatherGPT — Stage 2: Streamlit Chat UI
========================================

This file wraps the Stage 1 CLI logic into a beautiful chat interface
using Streamlit. The entire tool-calling pipeline is preserved exactly:
  User types question → LLM parses intent → get_weather tool → real API
  data → LLM narrates the answer → displayed in chat bubble

HOW THE UI WORKS:
  - st.chat_input()   → The text box at the bottom where user types
  - st.chat_message() → The chat bubbles (user = right, assistant = left)
  - st.session_state  → Remembers all messages across reruns (Streamlit
                         reruns the whole script on every interaction)
  - st.spinner()      → Shows "Fetching weather data..." while API calls run
  - st.sidebar        → Left panel with app info and settings

TO RUN LOCALLY:
  streamlit run app.py

CRITICAL RULE (same as Stage 1):
  The LLM NEVER invents weather numbers.
  All data comes from Open-Meteo API via the get_weather tool.

Author: WeatherGPT Team (SIH 2026)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load API keys from .env file (override=True ensures .env wins over stale
# terminal environment variables)
load_dotenv(override=True)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")


# ============================================================================
# WMO WEATHER CODE → HUMAN-READABLE STRING
# ============================================================================
# Open-Meteo returns weather conditions as numeric WMO standard codes.
# This dictionary converts those codes into descriptions the LLM can use.

WMO_WEATHER_CODES = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
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
# SYSTEM PROMPT — The "never invent numbers" rule for the LLM
# ============================================================================

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
• You may add helpful context (e.g., "that's quite warm for September") but every weather number must come from the tool result

━━━ WHAT YOU CAN DO ━━━
• Current weather for any city worldwide
• 3-day weather forecast (today + next 2 days)
• Rain predictions ("will it rain tomorrow in Chennai?")
• Farming/agriculture advice based on real weather data
• Travel weather checks ("what should I pack for Manali?")

━━━ ANSWERING STYLE ━━━
• Be warm, conversational, and helpful
• Always mention the city name in your answer
• Include specific numbers: temperature, humidity, rain chance, wind speed
• Keep answers concise but informative
• If the user greets you or asks a non-weather question, respond politely
• Mention the date/day when discussing forecasts
"""


# ============================================================================
# WEATHER TOOL — Fetches REAL data from Open-Meteo (free, no API key needed)
# ============================================================================

def get_weather(city_name: str) -> str:
    """
    Fetches real weather data for a city from Open-Meteo.

    Two-step process:
      1. Geocoding: city name → latitude/longitude
      2. Forecast:  lat/lon  → current weather + 3-day forecast

    Returns a JSON string the LLM will read and narrate.
    """

    # Step 1: Geocoding
    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if "results" not in geo_data or len(geo_data["results"]) == 0:
            return json.dumps({"error": True,
                               "message": f"City '{city_name}' not found. Try a nearby major city."})

        loc       = geo_data["results"][0]
        latitude  = loc["latitude"]
        longitude = loc["longitude"]
        resolved  = loc.get("name", city_name)
        country   = loc.get("country", "Unknown")
        state     = loc.get("admin1", "")

    except requests.exceptions.Timeout:
        return json.dumps({"error": True, "message": "Geocoding timed out. Try again."})
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": True, "message": "No internet connection."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": True, "message": f"Geocoding error: {e}"})

    # Step 2: Fetch weather forecast
    try:
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":  latitude,
                "longitude": longitude,
                "current": ",".join([
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "weather_code", "wind_speed_10m",
                    "wind_direction_10m", "cloud_cover",
                ]),
                "daily": ",".join([
                    "weather_code", "temperature_2m_max", "temperature_2m_min",
                    "precipitation_sum", "precipitation_probability_max",
                    "sunrise", "sunset", "uv_index_max", "wind_speed_10m_max",
                ]),
                "timezone":     "auto",
                "forecast_days": 3,
            },
            timeout=10,
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

    except requests.exceptions.Timeout:
        return json.dumps({"error": True, "message": "Weather fetch timed out. Try again."})
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": True, "message": "No internet connection."})
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": True, "message": f"Weather fetch error: {e}"})

    # Step 3: Structure the result
    current = weather_data.get("current", {})
    daily   = weather_data.get("daily",   {})

    cur_code      = current.get("weather_code", -1)
    cur_condition = WMO_WEATHER_CODES.get(cur_code, f"Unknown (code {cur_code})")

    result = {
        "source": "Open-Meteo API (real data, not AI-generated)",
        "location": {
            "city": resolved, "state": state, "country": country,
            "coordinates": f"{latitude:.4f}°N, {longitude:.4f}°E",
            "timezone": weather_data.get("timezone", "Unknown"),
        },
        "current_weather": {
            "temperature_celsius":    current.get("temperature_2m"),
            "feels_like_celsius":     current.get("apparent_temperature"),
            "humidity_percent":       current.get("relative_humidity_2m"),
            "precipitation_mm":       current.get("precipitation"),
            "wind_speed_kmh":         current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "cloud_cover_percent":    current.get("cloud_cover"),
            "condition":              cur_condition,
            "observation_time":       current.get("time", "Unknown"),
        },
        "three_day_forecast": [],
    }

    dates = daily.get("time", [])
    for i in range(len(dates)):
        def safe(key, idx=i):
            arr = daily.get(key, [])
            return arr[idx] if idx < len(arr) else None

        code      = safe("weather_code")
        condition = WMO_WEATHER_CODES.get(code, "Unknown") if code is not None else "Unknown"
        result["three_day_forecast"].append({
            "date":                  dates[i],
            "condition":             condition,
            "max_temp_celsius":      safe("temperature_2m_max"),
            "min_temp_celsius":      safe("temperature_2m_min"),
            "precipitation_total_mm":safe("precipitation_sum"),
            "rain_chance_percent":   safe("precipitation_probability_max"),
            "max_wind_speed_kmh":    safe("wind_speed_10m_max"),
            "uv_index_max":          safe("uv_index_max"),
            "sunrise":               safe("sunrise"),
            "sunset":                safe("sunset"),
        })

    return json.dumps(result, indent=2)


# ============================================================================
# TOOL DEFINITIONS — Formatted for Anthropic and OpenAI APIs
# ============================================================================

TOOL_DESCRIPTION = (
    "Fetches real-time weather data and a 3-day forecast for any city worldwide. "
    "Returns current temperature, humidity, wind, conditions, and daily forecasts "
    "with rain probability. Call this tool whenever the user asks about weather, "
    "temperature, rain, or climate for a specific location."
)

ANTHROPIC_TOOLS = [{
    "name": "get_weather",
    "description": TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "city_name": {
                "type": "string",
                "description": "Name of the city, e.g. 'Mumbai', 'Vizianagaram', 'Chennai'."
            }
        },
        "required": ["city_name"],
    },
}]

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "Name of the city, e.g. 'Mumbai', 'Vizianagaram', 'Chennai'."
                }
            },
            "required": ["city_name"],
        },
    },
}]

AVAILABLE_TOOLS = {"get_weather": get_weather}


# ============================================================================
# LLM CHAT FUNCTIONS — Tool-calling logic (same as CLI version)
# ============================================================================

def chat_with_anthropic(conversation_history: list, status_callback=None) -> str:
    """
    Sends conversation to Anthropic Claude, handles tool calling,
    and returns the final text response.

    status_callback: optional function(str) called to update loading message
    """
    import anthropic
    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=ANTHROPIC_TOOLS,
        messages=conversation_history,
    )

    if response.stop_reason == "tool_use":
        # Claude wants to call a weather tool
        conversation_history.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if status_callback:
                    status_callback(f"🔍 Fetching real weather data for **{block.input.get('city_name', '...')}**...")

                result = AVAILABLE_TOOLS[block.name](**block.input) \
                         if block.name in AVAILABLE_TOOLS \
                         else json.dumps({"error": f"Unknown tool: {block.name}"})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        conversation_history.append({"role": "user", "content": tool_results})

        if status_callback:
            status_callback("✍️ Generating your weather summary...")

        final = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=ANTHROPIC_TOOLS,
            messages=conversation_history,
        )

        text = "".join(b.text for b in final.content if hasattr(b, "text"))
        conversation_history.append({"role": "assistant", "content": final.content})
        return text

    else:
        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        conversation_history.append({"role": "assistant", "content": response.content})
        return text


def chat_with_openai(conversation_history: list, status_callback=None) -> str:
    """
    Sends conversation to OpenAI GPT, handles tool calling,
    and returns the final text response.
    """
    from openai import OpenAI
    client   = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=1024, tools=OPENAI_TOOLS, messages=messages
    )
    message = response.choices[0].message

    if message.tool_calls:
        conversation_history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ],
        })

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            if status_callback:
                status_callback(f"🔍 Fetching real weather data for **{args.get('city_name', '...')}**...")

            result = AVAILABLE_TOOLS[tc.function.name](**args) \
                     if tc.function.name in AVAILABLE_TOOLS \
                     else json.dumps({"error": f"Unknown tool: {tc.function.name}"})

            conversation_history.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if status_callback:
            status_callback("✍️ Generating your weather summary...")

        messages  = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        final     = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1024, tools=OPENAI_TOOLS, messages=messages
        )
        text = final.choices[0].message.content or ""
        conversation_history.append({"role": "assistant", "content": text})
        return text

    else:
        text = message.content or ""
        conversation_history.append({"role": "assistant", "content": text})
        return text


# ============================================================================
# DETECT LLM PROVIDER
# ============================================================================

def get_chat_function():
    """Returns the correct chat function based on available API key."""
    if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("your-"):
        return chat_with_anthropic, "Anthropic Claude"
    elif OPENAI_API_KEY and not OPENAI_API_KEY.startswith("your-"):
        return chat_with_openai, "OpenAI GPT"
    else:
        return None, None


# ============================================================================
# STREAMLIT PAGE CONFIG — Must be the FIRST Streamlit call in the file
# ============================================================================

st.set_page_config(
    page_title="WeatherGPT — AI Weather Assistant",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS — Premium look with smooth animations
# ============================================================================

st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background gradient */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 12px;
    padding: 4px 8px;
    animation: fadeInUp 0.3s ease;
}

/* Fade-in animation for messages */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Chat input box */
[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Title */
h1 {
    background: linear-gradient(90deg, #74ebd5, #acb6e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2rem !important;
    margin-bottom: 0.25rem !important;
}

/* Subtitle */
h1 + p {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.9rem !important;
}

/* General text */
p, li, span, label {
    color: rgba(255, 255, 255, 0.85) !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #74ebd5 !important;
}

/* Example question buttons */
.stButton > button {
    width: 100%;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.75rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: rgba(116, 235, 213, 0.12) !important;
    border-color: rgba(116, 235, 213, 0.35) !important;
    transform: translateX(3px) !important;
}

/* Info/warning boxes */
.stAlert {
    border-radius: 10px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🌤️ WeatherGPT")
    st.markdown("**AI Weather Assistant for India**")
    st.markdown("---")

    # Check API key status
    chat_fn, provider_name = get_chat_function()

    if chat_fn:
        st.success(f"✅ Connected to {provider_name}")
    else:
        st.error("❌ No API key found")
        st.markdown("""
        **To fix:**
        Open your `.env` file and paste your API key:
        ```
        ANTHROPIC_API_KEY=sk-ant-...
        ```
        """)

    st.markdown("---")

    # App info
    st.markdown("""
    **How it works:**
    1. You ask a weather question
    2. AI understands your intent
    3. Real data fetched from Open-Meteo
    4. AI explains the data in plain language

    > 🔒 **Data Integrity:**
    > All weather numbers come from
    > the Open-Meteo API — never
    > invented by the AI.
    """)

    st.markdown("---")

    # Example questions as clickable buttons
    st.markdown("**💬 Try asking:**")
    example_questions = [
        "Will it rain in Vizianagaram tomorrow?",
        "What's the weather in Mumbai right now?",
        "Should I carry an umbrella in Chennai today?",
        "Is it good weather to sow paddy in Guntur?",
        "What's the 3-day forecast for New Delhi?",
        "How hot is it in Hyderabad today?",
    ]

    # Store which example was clicked (if any)
    if "example_clicked" not in st.session_state:
        st.session_state.example_clicked = None

    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}"):
            st.session_state.example_clicked = q

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:rgba(255,255,255,0.35);'>
    WeatherGPT · SIH 2026<br>
    Data: Open-Meteo API (free, real-time)<br>
    Forecast: Current + 3 days
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# Title
st.markdown("# 🌤️ WeatherGPT")
st.markdown("Your AI weather assistant — powered by real data, never guesswork.")
st.markdown("---")

# ---- Session state initialisation ----
# Streamlit reruns the entire script on every user interaction.
# st.session_state persists data between those reruns (like a memory).

if "messages" not in st.session_state:
    # messages = list of {"role": "user"/"assistant", "content": str}
    # This is what we display in the chat bubbles.
    st.session_state.messages = []

if "history" not in st.session_state:
    # history = list of dicts in the exact format the LLM API expects.
    # For Anthropic, content can be a list of blocks (not just a string).
    # This is kept separate from `messages` so we can display cleanly.
    st.session_state.history = []

# ---- Check for API key ----
chat_fn, provider_name = get_chat_function()

if not chat_fn:
    # Show a friendly error — don't crash
    st.error(
        "⚠️ **No valid API key found.** "
        "Open your `.env` file and paste your Anthropic or OpenAI API key. "
        "Then restart the app with `streamlit run app.py`."
    )
    st.stop()  # Stop rendering — don't show the chat input

# ---- Render existing chat history ----
# Every time the page reruns, we re-draw all previous messages from session_state.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌤️"):
        st.markdown(msg["content"])

# ---- Handle example question button clicks ----
# If a sidebar button was clicked, inject it as the user's next input.
prompt_from_button = st.session_state.pop("example_clicked", None)

# ---- Chat input box ----
# st.chat_input() sits fixed at the bottom of the page.
# It returns the user's text when they press Enter, otherwise None.
user_input = st.chat_input("Ask me about the weather anywhere... 🌍") or prompt_from_button

if user_input:
    # 1️⃣ Show the user's message in a chat bubble immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2️⃣ Save user message to display history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 3️⃣ Add to LLM conversation history (used in API call)
    st.session_state.history.append({"role": "user", "content": user_input})

    # 4️⃣ Call the LLM with a loading spinner
    with st.chat_message("assistant", avatar="🌤️"):
        # The spinner shows while we wait for API responses
        with st.spinner("Thinking..."):

            # status_text holds a small st.empty() widget that we update
            # during the tool call to show progress steps to the user
            status_placeholder = st.empty()

            def update_status(msg: str):
                """Called from inside the LLM function to update the spinner label."""
                status_placeholder.markdown(
                    f"<div style='color:rgba(255,255,255,0.5); font-size:0.85rem;'>{msg}</div>",
                    unsafe_allow_html=True
                )

            try:
                # ▶ This is the core pipeline call — same logic as CLI version
                response_text = chat_fn(
                    st.session_state.history,
                    status_callback=update_status
                )
            except Exception as e:
                error_msg = str(e)
                # Friendly error messages based on error type
                if "401" in error_msg or "auth" in error_msg.lower() or "invalid" in error_msg.lower():
                    response_text = (
                        "❌ **Authentication failed.** Your API key appears to be invalid or expired. "
                        "Please update it in your `.env` file and restart the app."
                    )
                elif "429" in error_msg or "rate" in error_msg.lower():
                    response_text = (
                        "⏳ **Rate limit reached.** Please wait a moment and try again."
                    )
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    response_text = (
                        "📡 **Network error.** Please check your internet connection and try again."
                    )
                else:
                    response_text = (
                        f"⚠️ **Something went wrong:** {error_msg}\n\n"
                        "Please try again. If the problem continues, restart the app."
                    )

            # Clear the status message once done
            status_placeholder.empty()

        # 5️⃣ Display the assistant's response
        st.markdown(response_text)

    # 6️⃣ Save assistant response to display history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Note: st.session_state.history was already updated inside chat_fn
    # (both the user message AND the assistant response are appended there)

# ---- Empty state message (shown when no messages yet) ----
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 3rem 1rem; color:rgba(255,255,255,0.3);'>
        <div style='font-size: 4rem;'>🌤️</div>
        <div style='font-size: 1.1rem; margin-top: 1rem;'>
            Ask me anything about the weather!
        </div>
        <div style='font-size: 0.85rem; margin-top: 0.5rem;'>
            Try: "Will it rain in Vizianagaram tomorrow?"
        </div>
    </div>
    """, unsafe_allow_html=True)
