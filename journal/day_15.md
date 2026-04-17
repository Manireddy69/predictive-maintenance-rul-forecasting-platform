# Day 15

Today was where the project finally got a user-facing surface.

Up to this point, the repo had become much stronger technically.

But it still mostly communicated through:

- scripts
- CSV files
- JSON summaries
- experiment folders

That is fine for development.

It is not the best way to show the system as a coherent product.

## What I worked on

- built a Streamlit multipage application skeleton
- added an overview page for schedule health and sensitivity outputs
- added an equipment detail page for unit-level forecast inspection
- added an alerts configuration page
- added a reports page for schedule and sensitivity downloads
- added a shared data-loading module that reads the latest saved artifacts or falls back to sample data
- added deployment support for direct Streamlit runs, Docker, and hosted Streamlit deployment

## What got added

The dashboard layer now lives in:

- `app/streamlit_app.py`
- `app/dashboard_data.py`
- `app/pages/1_Overview.py`
- `app/pages/2_Equipment_Detail.py`
- `app/pages/3_Alerts_Configuration.py`
- `app/pages/4_Reports.py`

The deployment support now includes:

- `.streamlit/config.toml`
- `deploy/streamlit/Dockerfile`
- `requirements-streamlit.txt`
- `.dockerignore`

## Why this day mattered

The dashboard is not there to make the repo look prettier.

It matters because the project now has enough real artifacts to deserve a single inspection surface:

- latest forecast outputs
- latest schedule outputs
- sensitivity analysis
- report downloads

Once those exist, a dashboard becomes a practical interface, not just decoration.

## The hosted deployment issue

The interesting problem today was not the UI widgets.

It was deployment behavior.

The app worked locally when launched from the repo root.

But in hosted Streamlit deployment, `app/streamlit_app.py` runs in a different import context.

That caused the import issue around:

- `from app.dashboard_data import ...`

So I changed the dashboard files to load the app directory directly onto `sys.path` and import `dashboard_data` from there.

That made the hosted deployment path behave properly instead of only the local run path.

## What I checked

I verified the dashboard locally with:

- `streamlit run app/streamlit_app.py`

I also prepared and verified the Docker path:

- built the image successfully as `logicveda-streamlit`
- ran the container successfully
- confirmed the app responded on `http://localhost:8501`
- confirmed the HTTP endpoint returned `200`

On top of that, the hosted Streamlit deployment was fixed and confirmed running properly after the import patch.

## Why the fallback sample data mattered

One quiet but important decision was to make the dashboard usable even when fresh experiment artifacts are not present.

That means the UI can still launch and show a coherent structure instead of dying immediately on missing files.

This is especially useful for:

- first deployment
- demos on a clean environment
- hosted platforms where not every local artifact is present

## What I learned

- dashboard work becomes much cleaner once the upstream artifacts are already structured
- Streamlit deployment issues are often import-path issues rather than UI issues
- Docker verification is worth doing because local Python success does not guarantee container success
- fallback data is a good product decision for a project-stage dashboard

## What still feels shaky

- how rich the dashboard should become before it starts hiding weaknesses behind presentation
- whether alerts should remain a draft configuration page or become a persisted rule system in the next iteration
- how much of the forecasting and scheduling pipeline should eventually be triggerable directly from the UI

## Mistakes or traps

- building the dashboard before the artifact structure is stable
- assuming local import behavior will match hosted Streamlit behavior
- coupling the UI too tightly to one exact artifact folder layout without safe fallbacks

## What exists now

- a working Streamlit multipage dashboard
- a shared dashboard data layer
- a working Docker deployment path
- a working hosted Streamlit deployment

## Day 15 conclusion

Day 15 is complete.

The project now has a usable inspection and demo surface on top of the forecasting and scheduling pipeline.

That is important because the repo can now communicate the system through:

- metrics
- artifacts
- optimization outputs
- an actual UI

instead of only through code and folders.

## Next move

- clean the repo for final presentation
- collect screenshots and a concise demo flow
- write the polished report that explains the forecasting, scheduling, and deployment story clearly
