# 🌤️ WeatherGPT — Conversational Weather Intelligence Platform

> **Smart India Hackathon (SIH) 2026**
> Multilingual AI Weather Assistant for Farmers, Citizens & Disaster Management.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![GitHub License](https://img.shields.io/badge/SIH-2026-blue)](https://github.com/nelliramyasri405-sys/WeatherGPT)

---

## 🎯 Project Overview

**WeatherGPT** is a state-of-the-art conversational AI platform that delivers real-time weather forecasting, rain radar insights, agricultural advisories, and disaster alerts in multiple languages (**English, Telugu, Hindi**).

### 🔒 Core Credibility Rule (Zero Hallucinations)
> **The AI NEVER invents weather numbers.**  
> 100% of weather metrics (temperature, rainfall sum, humidity, wind speed, UV index) come directly from live verified [Open-Meteo API](https://open-meteo.com/) responses. The LLM serves exclusively as the natural language interpretation layer.

---

## ✨ Features

- 🌐 **Multilingual Voice & Text Support:** Seamless switching between **English**, **Telugu (తెలుగు)**, and **Hindi (हिंदी)** script.
- 🔊 **Voice Audio Assistant:** Instant spoken audio advisories generated via gTTS.
- 🌾 **Farmer Advisory Engine:** Tailored agricultural advice for crop sowing, harvesting risks, and pest impact based on live weather data.
- 🚨 **Emergency Alert Simulator:** Live simulation of NDMA/IMD Severe Weather Warnings (Cyclones & Heatwaves).
- 🎨 **Glassmorphic Interactive UI:** Custom fluid particle cursor, touch ripples, quick-city shortcuts, and atmospheric weather background.
- ⚡ **High-Speed Caching:** `@st.cache_data` enabled for instant responses and low latency.

---

## 🛠️ Tech Stack

- **Frontend & Web Framework:** [Streamlit](https://streamlit.io/) (Python)
- **AI Brain / LLM Engine:** [Groq Cloud](https://console.groq.com/) (`qwen/qwen3.6-27b`), Anthropic Claude 3.5 Sonnet, OpenAI GPT-4o-mini
- **Weather Data API:** [Open-Meteo Global Forecast API](https://open-meteo.com/) (100% free, no key needed)
- **Speech Synthesis:** `gTTS` (Google Text-to-Speech)
- **Styling:** Custom CSS Glassmorphism + Google Fonts (`Outfit`, `Inter`, `Noto Sans Telugu`, `Noto Sans Devanagari`)

---

## 🚀 How to Deploy (Step-by-Step Guide)

### Option 1: Deploy on Streamlit Community Cloud (Recommended & 100% FREE)

Deploying WeatherGPT on Streamlit Cloud takes less than 3 minutes:

1. **Push your code to GitHub:** (Already complete!)
   Repository: `https://github.com/nelliramyasri405-sys/WeatherGPT`

2. **Sign in to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with your GitHub account.

3. **Create a New App:**
   - Click **"New app"** -> **"Use existing repo"**.
   - Select your Repository: `nelliramyasri405-sys/WeatherGPT`
   - Select Branch: `main`
   - Main file path: `app.py`

4. **Add Your API Secrets:**
   - Click **"Advanced settings..."** or go to your app settings -> **Secrets**.
   - Paste your Groq API key (or Anthropic/OpenAI key):
     ```toml
     GROQ_API_KEY = "gsk_your_real_groq_key_here"
     ```
   - Click **Save**.

5. **Deploy!**
   - Click **"Deploy!"**. Your app will build automatically and give you a public live URL (e.g. `https://weathergpt.streamlit.app`).

---

### Option 2: Run Locally on Your Computer

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nelliramyasri405-sys/WeatherGPT.git
   cd WeatherGPT
   ```

2. **Create & Activate a Virtual Environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   - Create a `.env` file in the project root:
     ```env
     GROQ_API_KEY=gsk_your_groq_api_key_here
     ```

5. **Launch the Application:**
   - **Streamlit Web UI:**
     ```bash
     streamlit run app.py
     ```
     *Open `http://localhost:8501` in your browser.*

   - **CLI Mode (Terminal):**
     ```bash
     python weathergpt_core.py
     ```

---

## 📁 Repository Structure

```
WeatherGPT/
├── app.py                # Main Streamlit Web Application
├── weathergpt_core.py    # CLI Core Engine & LLM pipeline
├── requirements.txt      # Python dependencies
├── .env                  # API keys configuration (gitignored)
├── .env.example          # Template for environment variables
├── .gitignore            # Git protection rules
├── assets/               # High-resolution weather assets & backgrounds
│   ├── weather_bg.jpg
│   ├── farmer_advisory.jpg
│   └── disaster_radar.jpg
└── README.md             # Project documentation & deployment guide
```

---

## 👥 Team & License

- **Developed for:** Smart India Hackathon (SIH 2026)
- **License:** Open-source for educational and demonstration purposes.
