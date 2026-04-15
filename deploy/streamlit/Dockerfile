FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /workspace

COPY requirements-streamlit.txt ./requirements-streamlit.txt
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY app ./app
COPY .streamlit ./.streamlit
COPY Data/experiments ./Data/experiments

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
