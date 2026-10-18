# Deployment (free)

All options use the Groq free tier for the LLM. No paid services are involved.

## Option 1: Hugging Face Spaces (Docker, recommended)

The default container command runs the Streamlit UI with the agent in-process, so
a single free CPU Space is enough.

1. Create a Space: <https://huggingface.co/new-space>. Choose **Docker**, blank template,
   **CPU basic (free)**.
2. In the Space **Settings → Variables and secrets**, add the secret `GROQ_API_KEY`.
   Optional: `LANGSMITH_TRACING=true` and the secret `LANGSMITH_API_KEY`.
3. Deploy, either:
   * **From GitHub Actions** (keeps GitHub as the source of truth): add the repo secret
     `HF_TOKEN` (a Hugging Face *write* token) and the repo variable `HF_SPACE`
     (`username/space-name`), then run **Actions → Deploy to Hugging Face Space**.
     The workflow swaps in `deploy/huggingface/README.md`, whose front matter tells
     HF to use the Docker SDK on port 7860.
   * **Or manually**: `git clone` the Space, copy the repo in, replace `README.md` with
     `deploy/huggingface/README.md`, and push.

Notes:
* Free Spaces sleep after inactivity; the first request after waking takes ~30s.
* Disk is ephemeral: conversation memory (`data/checkpoints.sqlite`) resets on restart.
  The Chinook database is baked into the image at build time.

## Option 2: Streamlit Community Cloud

1. Push the repo to GitHub.
2. At <https://share.streamlit.io> create an app: main file
   `src/sql_analyst/ui/app.py`, and in **Advanced settings** pick **Python 3.11**.
3. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   (Root-level Streamlit secrets are exposed as environment variables, which the
   settings loader reads.)

Dependencies install from `requirements.txt`, generated from the lockfile with
`uv export --no-dev --no-hashes --no-header -o requirements.txt`; CI fails if it drifts. The
Chinook database downloads automatically (checksum-verified) on first start.

## Option 3: Local Docker / docker-compose

```bash
cp .env.example .env            # add GROQ_API_KEY
docker compose up --build       # UI :8501 -> API :8000
```

Compose runs the API and UI as separate services from one image; the UI reaches
the API through `API_URL`. Conversation memory is kept in a named volume.

Single container (same as HF Spaces):

```bash
docker build -t sql-analyst-agent .
docker run --env-file .env -p 7860:7860 sql-analyst-agent
```
