# Deployment Links

## Public Links

| Target | Status | URL |
| --- | --- | --- |
| GitHub repository | Available | https://github.com/Manireddy69/predictive-maintenance-rul-forecasting-platform |
| Static report site via raw.githack | Verified live | https://raw.githack.com/Manireddy69/predictive-maintenance-rul-forecasting-platform/gh-pages/index.html |
| Static report site via jsDelivr | Verified live | https://cdn.jsdelivr.net/gh/Manireddy69/predictive-maintenance-rul-forecasting-platform@gh-pages/index.html |
| GitHub Pages report site | Workflow succeeded; repo Pages setting still returns 404 | https://manireddy69.github.io/predictive-maintenance-rul-forecasting-platform/ |
| Streamlit dashboard | Requires Streamlit Cloud login | Add URL after deployment |
| Render dashboard | Blueprint configured | Add URL after deployment |
| Railway dashboard | Configured | Add URL after deployment |
| Hugging Face Space | Not created from this workspace | Add URL after deployment |

## Local Links

| Service | URL |
| --- | --- |
| Streamlit dashboard | http://localhost:8501 |
| Airflow webserver | http://localhost:8080 |

## Deployment Notes

The GitHub Pages site publishes the final report, screenshots, compliance evidence, and demo assets from the `reports/` folder.

The `gh-pages` branch is published. To enable the canonical GitHub Pages URL, open:

```text
Repository Settings -> Pages -> Build and deployment -> Source: Deploy from a branch -> Branch: gh-pages / root -> Save
```

After GitHub finishes provisioning, the canonical URL should be:

```text
https://manireddy69.github.io/predictive-maintenance-rul-forecasting-platform/
```

The Streamlit app entrypoint for all app-hosting platforms is:

```text
app/streamlit_app.py
```

The app can be deployed with Docker using:

```bash
docker build -t predictive-maintenance-rul-dashboard .
docker run --rm -p 8501:8501 predictive-maintenance-rul-dashboard
```
