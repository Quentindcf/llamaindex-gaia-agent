---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# 🧠 GAIA Agent — Level 1 Benchmark

A modular agent built with [LlamaIndex](https://llamaindex.ai) to tackle the first level of the [GAIA Benchmark](https://github.com/GAIA-benchmark/GAIA).

## 🎯 Goal

Achieve at least **30% accuracy** on GAIA Level 1 using an autonomous agent that can:
- Parse and reason about multimodal inputs (images, audio, video)
- Use external tools (transcription, captioning, web search)
- Chain reasoning steps together using a Planner–Executor architecture

## ⚙️ Frameworks

- LlamaIndex
- OpenAI / Mistral / Anthropic (LLM backend)
- Whisper / BLIP / web search tools

## 🧰 Features

- Modular tool system (plug-and-play)
- Logging for each agent run
- Agent memory / scratchpad
- Evaluator for automatic scoring

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/gaia-agent.git
cd gaia-agent

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Run agent on a sample question
python run_agent.py --question_id q_0001
```

## 📂 Repo Structure
```
gaia-agent/
├── tools/              # Tool wrappers (e.g. whisper.py, blip.py)
├── agent/              # Agent logic
├── data/               # GAIA samples and outputs
├── run_agent.py        # CLI entrypoint
├── eval/               # Scoring logic
├── README.md
└── requirements.txt
```

## 📜 License

MIT

## 🧱 Environment Setup

### `requirements.txt` (initial draft)

```txt
llama-index>=0.10.40
openai
tavily-search  # optional for web search
transformers
torch
accelerate
Pillow
whisper
opencv-python
python-dotenv
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

