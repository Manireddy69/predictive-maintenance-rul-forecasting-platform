# Live Demo Verification

## Local Demo

Status: verified.

Command:

```powershell
Invoke-WebRequest http://localhost:8501 -UseBasicParsing -TimeoutSec 10
```

Result:

```text
HTTP 200
```

Local URL:

```text
http://localhost:8501
```

## Public Demo

Status: not verified from this workspace.

Reason:

- no public Streamlit/Hugging Face/Vercel URL is stored in the repository
- web search did not locate a public app for this repository
- publishing requires an authenticated deployment account

Required final action:

Deploy `app/streamlit_app.py` to Streamlit Community Cloud or another public host, then add the URL here and test it from a logged-out browser.

## Deployment Entry Point

```text
app/streamlit_app.py
```

## Docker Path

```bash
docker build -f deploy/streamlit/Dockerfile -t logicveda-streamlit .
docker run --rm -p 8501:8501 logicveda-streamlit
```
