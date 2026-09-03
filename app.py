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

# Load API keys — supports both local (.env file) AND Streamlit Cloud (st.secrets)
# Locally:  Keys come from .env file via python-dotenv
# On Cloud: Keys come from Streamlit's "Secrets" settings (set in dashboard)
load_dotenv(override=True)

def get_api_key(key_name: str) -> str:
    """
    Gets an API key, checking multiple sources in priority order:
    1. Streamlit secrets (for Streamlit Cloud deployment)
    2. Environment variables / .env file (for local development)
    """
    # Try Streamlit secrets first (used on Streamlit Community Cloud)
    try:
        value = st.secrets.get(key_name, None)
        if value:
            return value
    except Exception:
        pass  # st.secrets not available or not configured

    # Fall back to environment variable / .env file
    return os.environ.get(key_name, "")

ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = get_api_key("OPENAI_API_KEY")
GROQ_API_KEY      = get_api_key("GROQ_API_KEY")


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

GROQ_API_KEY = get_api_key("GROQ_API_KEY")


def chat_with_groq(conversation_history: list, status_callback=None) -> str:
    """
    Sends conversation to Groq Cloud (High-speed free AI engine), handles tool calling,
    and returns the final text response.
    """
    from openai import OpenAI
    key = GROQ_API_KEY or (ANTHROPIC_API_KEY if ANTHROPIC_API_KEY.startswith("gsk_") else "")
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b", max_tokens=1024, tools=OPENAI_TOOLS, messages=messages
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

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        final = client.chat.completions.create(
            model="qwen/qwen3.6-27b", max_tokens=1024, tools=OPENAI_TOOLS, messages=messages
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
    if (GROQ_API_KEY and not GROQ_API_KEY.startswith("your-")) or ANTHROPIC_API_KEY.startswith("gsk_"):
        return chat_with_groq, "Groq AI (Free & Fast)"
    elif ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("your-"):
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
# VOICE TTS ASSISTANT HELPER (gTTS)
# ============================================================================

def generate_voice_audio(text_summary: str, lang_code: str = 'en') -> bytes:
    """Generates audio MP3 bytes from text using gTTS for accessibility."""
    try:
        from gtts import gTTS
        import io

        # Clean markdown formatting tags for natural speech
        clean_text = text_summary.replace("**", "").replace("*", "").replace("`", "").replace("#", "").replace(">", "")
        if len(clean_text) > 350:
            clean_text = clean_text[:350] + "..."
        
        tts = gTTS(text=clean_text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None


# Language Code Mapping
LANG_MAP = {
    "English": ("en", "Please respond in clear English."),
    "Hindi (हिंदी)": ("hi", "Please translate and respond in clear Hindi (हिंदी)."),
    "Telugu (తెలుగు)": ("te", "Please translate and respond in clear Telugu (తెలుగు)."),
    "Tamil (தமிழ்)": ("ta", "Please translate and respond in clear Tamil (தமிழ்)."),
    "Bengali (বাংলা)": ("bn", "Please translate and respond in clear Bengali (বাংলা)."),
}


# ============================================================================
# CUSTOM CSS — World-Class SIH 2026 Glassmorphic Interface
# ============================================================================

st.markdown("""
<style>
/* Import Google Fonts: Outfit for Titles, Inter for Body */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Base */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Custom Pointer Cursors */
html, body, button, input, select, textarea, [role="button"], a {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2300f2fe' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 3l7 18 3-7 7-3L3 3z'/%3E%3C/svg%3E"), auto !important;
}

button:hover, [role="button"]:hover, a:hover, select:hover {
    cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='%2300f2fe' stroke='%2374ebd5' stroke-width='1.5'%3E%3Ccircle cx='12' cy='12' r='8' fill-opacity='0.3'/%3E%3Cpath d='M3 3l7 18 3-7 7-3L3 3z'/%3E%3C/svg%3E"), pointer !important;
}

/* App Background: Rich Mesh Gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, #1a103c 0%, #0d0b1e 50%, #05040a 100%);
    min-height: 100vh;
}

/* Header Container */
.main-header {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    animation: fadeInDown 0.5s ease-out;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-15px); }
    to   { opacity: 1; transform: translateY(0); }
}

.brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #00c6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0 !important;
    display: inline-block;
}

.brand-subtitle {
    color: rgba(255, 255, 255, 0.65);
    font-size: 0.95rem;
    margin-top: 0.3rem;
    font-weight: 400;
}

/* Category Badge Pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.2);
    border-radius: 30px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #74ebd5;
    margin-right: 6px;
    margin-top: 8px;
}

/* Emergency Alert Banner */
.alert-card {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
    animation: pulseAlert 2s infinite;
}

@keyframes pulseAlert {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
    70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

/* Chat Message Bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(16px) !important;
    margin-bottom: 14px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
    animation: fadeInUp 0.35s ease-out !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #4facfe !important;
    background: rgba(79, 172, 254, 0.06) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #00f2fe !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Chat Input Styling */
[data-testid="stChatInput"] textarea {
    background: rgba(20, 18, 45, 0.75) !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.3) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(12, 10, 28, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(25px) !important;
}

/* Prompt Card Buttons */
.stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.9rem !important;
    text-align: left !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 6px !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button:hover {
    background: rgba(0, 242, 254, 0.12) !important;
    border-color: rgba(0, 242, 254, 0.4) !important;
    color: #ffffff !important;
    transform: translateX(4px) scale(1.01) !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.15) !important;
}

/* Status Cards & Banners */
.status-connected {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 8px 14px;
    color: #34d399;
    font-size: 0.82rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Architecture Callout Box */
.arch-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 12px 14px;
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.65);
    margin-top: 10px;
}

/* Hero Feature Cards */
.hero-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}

.hero-card:hover {
    background: rgba(0, 242, 254, 0.05);
    border-color: rgba(0, 242, 254, 0.25);
    transform: translateY(-3px);
}

.hero-icon {
    font-size: 2.2rem;
    margin-bottom: 8px;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 4px;
}

.hero-desc {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.55);
}

/* Spinner Customisation */
.stSpinner > div {
    border-top-color: #00f2fe !important;
}
</style>

<script>
(function() {
    if (window.cursorInitialized) return;
    window.cursorInitialized = true;

    // Create glowing dot follower
    const dot = document.createElement('div');
    dot.id = 'glow-cursor-dot';
    dot.style.cssText = `
        position: fixed;
        width: 14px;
        height: 14px;
        background: radial-gradient(circle, #00f2fe 0%, #74ebd5 70%, transparent 100%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 999999;
        transform: translate(-50%, -50%);
        transition: transform 0.08s ease-out;
        box-shadow: 0 0 15px #00f2fe, 0 0 30px #00f2fe;
    `;
    document.body.appendChild(dot);

    let mouseX = -100, mouseY = -100;

    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        dot.style.left = mouseX + 'px';
        dot.style.top = mouseY + 'px';
    });

    // Touchscreen Ripple Effect
    window.addEventListener('touchstart', (e) => {
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            const ripple = document.createElement('div');
            ripple.style.cssText = `
                position: fixed;
                left: ${touch.clientX}px;
                top: ${touch.clientY}px;
                width: 12px;
                height: 12px;
                border: 2px solid #00f2fe;
                border-radius: 50%;
                pointer-events: none;
                z-index: 999999;
                transform: translate(-50%, -50%) scale(1);
                animation: touchRipple 0.5s ease-out forwards;
                box-shadow: 0 0 12px #00f2fe;
            `;
            document.body.appendChild(ripple);
            setTimeout(() => ripple.remove(), 500);
        }
    });

    const style = document.createElement('style');
    style.innerHTML = `
        @keyframes touchRipple {
            0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
            100% { transform: translate(-50%, -50%) scale(5); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
})();
</script>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom: 12px;'>
        <div style='font-family: "Outfit", sans-serif; font-size: 1.5rem; font-weight: 800; color: #ffffff;'>
            🌤️ WeatherGPT
        </div>
        <div style='font-size: 0.8rem; color: rgba(255,255,255,0.5);'>
            Smart India Hackathon (SIH 2026)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check API key status
    chat_fn, provider_name = get_chat_function()

    if chat_fn:
        st.markdown(f"""
        <div class="status-connected">
            <span style="font-size:10px;">🟢</span> {provider_name} Connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ No API Key Found")

    st.markdown("---")

    # Multilingual Language Selector
    selected_lang = st.selectbox(
        "🌐 Voice & Language Output",
        options=list(LANG_MAP.keys()),
        index=0
    )
    lang_code, lang_inst = LANG_MAP[selected_lang]

    # Voice TTS Toggle
    enable_voice = st.toggle("🔊 Enable Audio Speech Assistant", value=True)

    st.markdown("---")

    # Disaster Alert Scenario Trigger (SIH Judging Feature)
    with st.expander("🚨 Emergency Alert Simulator (SIH Demo)"):
        st.markdown("<div style='font-size:0.8rem; color:rgba(255,255,255,0.7); margin-bottom:8px;'>Demonstrates automated NDMA/IMD disaster alert dissemination:</div>", unsafe_allow_html=True)
        if st.button("⚡ Trigger Cyclone Alert Scenario"):
            st.session_state.active_alert = "CYCLONE"
        if st.button("⚡ Trigger Heatwave Alert Scenario"):
            st.session_state.active_alert = "HEATWAVE"
        if st.button("❌ Clear Active Alert"):
            st.session_state.active_alert = None

    st.markdown("---")

    # Architecture Callout
    st.markdown("""
    <div class="arch-box">
        <strong style="color:#74ebd5;">🔒 Core Credibility Rule:</strong><br>
        The LLM <em>never</em> invents forecast numbers. 100% of temperatures, rain, and wind metrics are fetched live from Open-Meteo meteorological APIs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:rgba(255,255,255,0.35); text-align:center;'>
        WeatherGPT · Smart India Hackathon 2026<br>
        Open-Meteo Real-Time Data Pipeline
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# Header Banner
st.markdown("""
<div class="main-header">
    <div class="brand-title">🌤️ WeatherGPT</div>
    <div class="brand-subtitle">Conversational Weather Intelligence Platform for Farmers, Disaster Managers & Citizens</div>
    <div>
        <span class="badge-pill">🌾 Agricultural Advisory</span>
        <span class="badge-pill">⛈️ Live Rain Radar</span>
        <span class="badge-pill">🔊 Multilingual Voice Assistant</span>
        <span class="badge-pill">🔒 Verified API Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Simulated Emergency Alert (if active)
if st.session_state.get("active_alert") == "CYCLONE":
    st.markdown("""
    <div class="alert-card">
        <div style="color: #ef4444; font-weight: 800; font-size: 1.1rem;">🚨 IMD SEVERE CYCLONE WARNING: COASTAL ANDHRA PRADESH & TAMIL NADU</div>
        <div style="color: rgba(255,255,255,0.85); font-size: 0.9rem; margin-top: 6px;">
            Sustained wind speeds of 85-95 km/h expected. Heavy to extremely heavy rainfall forecast across Coastal districts. Farmers advised to delay harvesting & secure stored grains immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.get("active_alert") == "HEATWAVE":
    st.markdown("""
    <div class="alert-card" style="background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.4);">
        <div style="color: #f59e0b; font-weight: 800; font-size: 1.1rem;">⚠️ IMD SEVERE HEATWAVE RED ALERT: TELANGANA & RAYALASEEMA</div>
        <div style="color: rgba(255,255,255,0.85); font-size: 0.9rem; margin-top: 6px;">
            Maximum temperatures expected to cross 44°C. Avoid outdoor farm activities between 11:00 AM and 3:30 PM. Ensure frequent crop irrigation.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Session state initialisation
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Check API key
chat_fn, provider_name = get_chat_function()

if not chat_fn:
    st.error(
        "⚠️ **No valid API key found.** "
        "Open your `.env` file and paste your Groq, Anthropic, or OpenAI API key."
    )
    st.stop()

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌤️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("audio_bytes"):
            st.audio(msg["audio_bytes"], format="audio/mp3")

# Handle example question clicks or quick city clicks
prompt_from_button = st.session_state.pop("example_clicked", None)

# Quick City Shortcuts Bar
st.markdown("<div style='font-size:0.85rem; font-weight:700; color:rgba(255,255,255,0.8); margin-bottom:6px;'>📍 Quick City Weather Search:</div>", unsafe_allow_html=True)
city_cols = st.columns(7)
quick_cities = ["Vizianagaram", "Visakhapatnam", "Guntur", "Chennai", "Hyderabad", "Delhi", "Mumbai"]
for idx, city in enumerate(quick_cities):
    with city_cols[idx]:
        if st.button(f"📍 {city}", key=f"quick_{city}"):
            st.session_state.example_clicked = f"What is the weather in {city} right now and 3-day forecast?"
            st.rerun()

# Chat input box
user_input = st.chat_input("Ask about weather, rain forecasts, or farming advice... 🌍") or prompt_from_button

if user_input:
    # Append language instructions if non-English
    llm_query = user_input
    if selected_lang != "English":
        llm_query = f"{user_input}\n\n[Instruction: {lang_inst}]"

    # 1. User bubble
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "user", "content": llm_query})

    # 2. Assistant response
    with st.chat_message("assistant", avatar="🌤️"):
        with st.spinner("Analyzing query & fetching live meteorological data..."):
            status_placeholder = st.empty()

            def update_status(msg: str):
                status_placeholder.markdown(
                    f"<div style='color:#74ebd5; font-size:0.85rem; padding:4px 0;'>{msg}</div>",
                    unsafe_allow_html=True
                )

            try:
                response_text = chat_fn(
                    st.session_state.history,
                    status_callback=update_status
                )
            except Exception as e:
                error_msg = str(e)
                if "credit balance" in error_msg.lower() or "balance" in error_msg.lower():
                    response_text = (
                        "💳 **Credit balance too low.** Your API account has 0 credits. "
                        "Please check your API dashboard or switch provider keys in `.env`."
                    )
                elif "401" in error_msg or "auth" in error_msg.lower() or "invalid" in error_msg.lower():
                    response_text = (
                        "❌ **Authentication failed.** API key invalid or expired. "
                        "Please check your `.env` file."
                    )
                elif "429" in error_msg or "rate" in error_msg.lower():
                    response_text = "⏳ **Rate limit reached.** Please wait a moment and try again."
                else:
                    response_text = f"⚠️ **Notice:** {error_msg}\n\nPlease try asking again."

            status_placeholder.empty()

        st.markdown(response_text)

        # Generate Audio Voice TTS if enabled
        audio_data = None
        if enable_voice:
            with st.spinner("🔊 Generating voice audio advisory..."):
                audio_data = generate_voice_audio(response_text, lang_code=lang_code)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "audio_bytes": audio_data
    })

# Domain Persona Tabs / Interactive Welcome Screen (when chat is empty)
if not st.session_state.messages:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌾 Farmer Advisory",
        "🌧️ Rain & Forecast",
        "🚨 Disaster Radar",
        "✈️ Aviation & Marine"
    ])

    with tab1:
        st.markdown("### 🌾 Crop Weather & Harvesting Guidance")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌾 Should I sow paddy this week in Vizianagaram?"):
                st.session_state.example_clicked = "Should I sow paddy this week in Vizianagaram?"
                st.rerun()
        with c2:
            if st.button("🌾 Is there rain risk for cotton harvesting in Guntur?"):
                st.session_state.example_clicked = "Is there rain risk for cotton harvesting in Guntur?"
                st.rerun()

    with tab2:
        st.markdown("### 🌧️ Real-Time Rain & Temperature Radar")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌧️ Will it rain in Chennai tomorrow?"):
                st.session_state.example_clicked = "Will it rain in Chennai tomorrow?"
                st.rerun()
        with c2:
            if st.button("🌡️ What is the 3-day forecast for New Delhi?"):
                st.session_state.example_clicked = "What is the 3-day forecast for New Delhi?"
                st.rerun()

    with tab3:
        st.markdown("### 🚨 Disaster Alert & Extreme Weather Briefing")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚡ Check cyclone & storm warnings for Visakhapatnam"):
                st.session_state.example_clicked = "Check cyclone & storm warnings for Visakhapatnam"
                st.rerun()
        with c2:
            if st.button("🔥 Heatwave advisory for Rayalaseema districts"):
                st.session_state.example_clicked = "Heatwave advisory for Rayalaseema districts"
                st.rerun()

    with tab4:
        st.markdown("### ✈️ Aviation, Marine & Coastal Advisory")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🌊 Coastal wind speed & wave state in Machilipatnam"):
                st.session_state.example_clicked = "Coastal wind speed & wave state in Machilipatnam"
                st.rerun()
        with c2:
            if st.button("✈️ Visibility & storm conditions for Hyderabad airport"):
                st.session_state.example_clicked = "Visibility & storm conditions for Hyderabad airport"
                st.rerun()

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌾\n**Farmer Advisory**\nCrop sowing & harvest advisories", key="hero_farmer"):
            st.session_state.example_clicked = "Give me a farmer advisory for crop sowing and rain risks in Vizianagaram"
            st.rerun()

    with col2:
        if st.button("🔊\n**Multilingual Audio**\nSpoken advisories in 5 languages", key="hero_audio"):
            st.session_state.example_clicked = "What is the 3-day weather forecast for Chennai?"
            st.rerun()

    with col3:
        if st.button("🔒\n**Zero Hallucinations**\nStrict verified Open-Meteo API data", key="hero_verified"):
            st.session_state.example_clicked = "Explain current weather in Mumbai with exact API data"
            st.rerun()

