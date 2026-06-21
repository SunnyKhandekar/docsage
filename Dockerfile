# DocSage — Dockerfile for Hugging Face Spaces (Docker SDK)
#
# HF Spaces Docker SDK notes:
#   - Port MUST be 7860 (HF routes all traffic to this port)
#   - GROQ_API_KEY is set as a Space secret; it arrives as a plain env var
#   - Do NOT run Docker locally on a 4GB RAM laptop — let HF build it server-side
#   - python-slim keeps the image small; no system-level ML libs needed

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal — PyMuPDF needs libmupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer cache: only rebuilds if
# requirements.txt changes, not if source code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose HF Spaces' required port
EXPOSE 7860

# Run Streamlit on the correct port with headless mode for containerised environments
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
