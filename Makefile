.PHONY: install data test lint format api ui

install:  ## Install dependencies (Python 3.11 via uv)
	uv sync

data:  ## Download the Chinook sample database
	uv run python data/download_chinook.py

test:  ## Run the test suite (no API keys needed)
	uv run pytest

lint:  ## Lint and check formatting
	uv run ruff check . && uv run ruff format --check .

format:  ## Auto-format and fix lint issues
	uv run ruff format . && uv run ruff check --fix .

api:  ## Run the FastAPI backend on :8000
	uv run uvicorn sql_analyst.api.app:app --reload --port 8000

ui:  ## Run the Streamlit UI on :8501 (in-process agent; set API_URL to use the API)
	uv run streamlit run src/sql_analyst/ui/app.py
