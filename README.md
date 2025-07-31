---
title: GAIA Agent – Final Assignment
emoji: 🧠
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# 🧠 GAIA Agent — Level 1 Benchmark

A modular multimodal agent built with [LlamaIndex](https://llamaindex.ai), designed to tackle Level 1 of the [GAIA Benchmark](https://github.com/GAIA-benchmark/GAIA).

## 🎯 Goal

Achieve at least **30% accuracy** on GAIA Level 1 using an autonomous FunctionAgent–based architecture that can:
- Parse and reason over multimodal inputs (images, video, audio, text)
- Use specialized tools (transcription, video analysis, Wikipedia, Arxiv, web search)
- Delegate tasks between agents in a planner–executor system

## ⚙️ Frameworks & Dependencies

- [LlamaIndex](https://llamaindex.ai)
- OpenAI (GPT-4.1)
- Whisper (via LlamaIndex)
- Tavily API for live web search
- yt-dlp, OpenCV for YouTube video handling
- Gradio (for Hugging Face UI)

## 🧰 Features

- 🔌 Modular tool system (plug-and-play)
- 🪄 Controller agent to route questions to the right expert
- 🔊 Audio transcription (file or YouTube)
- 🎥 YouTube video download and frame extraction
- 🌐 Web search + Wikipedia + Arxiv lookup
- ✅ Built-in logic to return concise `FINAL ANSWER`

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/llamaindex-gaia-agent.git
cd llamaindex-gaia-agent

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Add your environment variables
cp .env.example .env
# Edit .env to add your OpenAI and Tavily API keys

# 4. Run the agent (sample)
python app/main.py
```

## 📂 Folder Structure

```
llamaindex-gaia-agent/
├── agents/                 # Core agent logic and controller
├── tools/                 # Video, audio, and search tools
├── data/                  # Sample inputs and outputs
├── app.py                 # Gradio interface
├── .env.example           # Example environment variables
├── requirements.txt
└── README.md
```

## 🧪 .env Configuration

Create a .env file with your credentials:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly_...
```

These are loaded automatically by the agent.

## 📜 License

MIT