# VeriScript container: REST API (port 8000) and Streamlit UI (port 8501), CPU-only by default.
# Build after training so that models/ is populated:
#   docker build -t veriscript .
#   docker run -p 8000:8000 -p 8501:8501 veriscript
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf VERISCRIPT_DEVICE=cpu VERISCRIPT_API_URL=http://127.0.0.1:8000

WORKDIR /app
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY veriscript ./veriscript
RUN pip install . && python -m spacy download en_core_web_sm \
    && python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; T.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')"
COPY configs ./configs
COPY app ./app
COPY models ./models

EXPOSE 8000 8501
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
