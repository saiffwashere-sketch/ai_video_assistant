# AI Video Assistant

An app that takes a video, transcribes it, summarizes it, and lets you ask questions about it using RAG.

## Features

- Extract audio from videos
- Transcribe speech to text
- Summarize the content
- Ask questions about the video

## Setup

1. Clone the repo
   git clone https://github.com/saiffwashere-sketch/ai_video_assistant.git
   cd ai_video_assistant

2. Create a virtual environment and install dependencies
   uv venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt

3. Create a `.env` file with your API key
   GROQ_API_KEY=your_key_here

## Run

python app.py

## Tech stack

- Python, LangChain, Groq, ChromaDB
