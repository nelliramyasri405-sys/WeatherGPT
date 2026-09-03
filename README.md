# 🌤️ WeatherGPT — AI Weather Assistant

> **Smart India Hackathon (SIH) 2026**
> Conversational AI platform for weather forecasting, alerts, and climate information.

---

## 🎯 What Is This?

WeatherGPT is a conversational AI that answers weather questions in natural language.
Ask it anything — *"Will it rain in Vizianagaram tomorrow?"* — and it gives you an
accurate, human-friendly answer powered by **real weather data**.

### Core Principle

> **The AI never invents weather numbers.**
> All temperatures, rainfall, humidity, and forecasts come from the Open-Meteo API.
> The AI's job is to understand your question and explain the real data clearly.

---

## 🏗️ Architecture (5-Stage Pipeline)

```
User question (text)
        ↓
LLM Layer #1 — Parses intent, extracts city name
        ↓
Weather API — Fetches REAL data from Open-Meteo (free, no API key)
        ↓
LLM Layer #2 — Converts raw data into natural language answer
        ↓
Delivery — Text response in terminal
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed ([download here](https://www.python.org/downloads/))
- An API key from **one** of:
  - [Anthropic Claude](https://console.anthropic.com/) (recommended)
  - [OpenAI GPT](https://platform.openai.com/api-keys)

### Step 1 — Clone / Download the Project

Download or clone this folder to your computer.

### Step 2 — Create a Virtual Environment

Open a terminal **in this project folder** and run:

```bash
# Create the virtual environment
python -m venv venv

# Activate it:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (CMD):
.\venv\Scripts\activate.bat

# On Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set Your API Key

Open the `.env` file in a text editor and replace the placeholder with your real key:

```env
# If using Anthropic Claude:
ANTHROPIC_API_KEY=sk-ant-your-real-key-here

# If using OpenAI GPT (uncomment and fill in):
# OPENAI_API_KEY=sk-your-real-key-here
```

> ⚠️ **NEVER** share your `.env` file, commit it to Git, or paste your key in `.env.example`.
> The `.gitignore` file protects `.env` from being uploaded.

### Step 5 — Run WeatherGPT

```bash
python weathergpt_core.py
```

Then just type your weather questions!

---

## 💬 Example Conversations

```
You: Will it rain in Chennai tomorrow?
🌤️ WeatherGPT: Based on the forecast for Chennai, there's a 72% chance of
rain tomorrow with moderate rain expected. The high will be 33°C and the
low 27°C. I'd recommend carrying an umbrella!

You: What about Mumbai?
🌤️ WeatherGPT: In Mumbai, tomorrow looks partly cloudy with a 45% chance
of rain. Temperatures will range from 26°C to 32°C...

You: Should I sow paddy this week in Guntur?
🌤️ WeatherGPT: Looking at the 3-day forecast for Guntur — temperatures
are between 28-36°C with low rain probability (15%). These are dry
conditions, so ensure irrigation is available if you proceed with sowing...
```

---

## 📁 Project Structure

```
weatherGPT forcasting/
├── weathergpt_core.py    # Main script — LLM + tool calling + Open-Meteo
├── requirements.txt      # Python dependencies
├── .env                  # YOUR API key (secret, never commit this)
├── .env.example          # Template showing what .env should look like
├── .gitignore            # Protects .env from being committed to Git
└── README.md             # This file
```

---

## 🔧 Tech Stack

| Layer            | Tool                          | Notes                                          |
|------------------|-------------------------------|-------------------------------------------------|
| LLM              | Anthropic Claude / OpenAI GPT | Understands questions, narrates answers          |
| Weather Data     | Open-Meteo API                | Free, no API key, real-time global forecasts     |
| Language         | Python 3.10+                  | Simple, beginner-friendly                        |
| Key Management   | python-dotenv + .env file     | Keeps API keys safe and out of code              |

---

## 🔒 Security Rules

1. **NEVER** put real API keys in `.env.example` or any code file
2. **ALWAYS** use the `.env` file for real keys (it's gitignored)
3. **NEVER** commit `.env` to Git — the `.gitignore` prevents this
4. If you accidentally expose a key, **regenerate it immediately** at your provider's dashboard

---

## 🗺️ Roadmap

- [x] **Stage 1** — CLI core loop (LLM + tool calling + Open-Meteo) ← *You are here*
- [ ] **Stage 2** — Streamlit chat UI
- [ ] **Stage 3** — Multilingual support (Hindi, Telugu, Tamil, Bengali)
- [ ] **Stage 4** — Simulated weather alerts (cyclone, flood, heatwave)
- [ ] **Stage 5** — Voice input/output (stretch goal)
- [ ] **Stage 6** — Polish, error handling, demo prep

---

## 👥 Team

WeatherGPT Team — Smart India Hackathon 2026

---

## 📄 License

Built for SIH 2026. For educational and demonstration purposes.
