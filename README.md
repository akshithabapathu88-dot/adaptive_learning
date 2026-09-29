# Adaptive Learning AI

This project is an adaptive learning system that generates personalized learning content, quizzes, assignments, and PDF reports based on student performance.

## Features

- Personalized learning content generation
- Adaptive quizzes and evaluation
- Assignment generation and scoring
- Student analytics and recommendations
- PDF report generation
- Vector-based retrieval for topic context

## Setup

1. Create a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Add your Groq API key:
   - Copy `.env.example` to `.env`
   - Set `GROQ_API_KEY` to your actual key
4. Run the app:
   ```powershell
   python main.py run
   ```

## Project Layout

- `main.py` — CLI entry point
- `streamlit_app.py` — root Streamlit app
- `frontend/streamlit_app.py` — frontend app
- `database/` — database schema and access layer
- `rag/` — retrieval and vector store logic
- `agents/` — AI-powered learning agents
- `pdf_generator/` — PDF generation

## Notes

The app expects a valid Groq API key to generate learning content, quizzes, and recommendations.
