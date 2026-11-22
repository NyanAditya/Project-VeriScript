# VeriScript public demo on Hugging Face Spaces (Docker SDK, CPU basic hardware).
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    VERISCRIPT_HOME=/home/user/app \
    VERISCRIPT_API_URL= \
    VERISCRIPT_DEMO=1 \
    MODEL_REPO=__MODEL_REPO__
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Bake the models into the image so that a restart after sleep does not re-download them.
RUN python download_models.py

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false", \
     "--server.maxUploadSize=10", "--browser.gatherUsageStats=false"]
