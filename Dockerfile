FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0

WORKDIR /app

# Dependências mínimas
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python
RUN pip install --no-cache-dir \
    psutil

# Instala Ollama (versão CPU)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copia scripts
COPY benchmark.py analyze.py ./

# Diretórios esperados
RUN mkdir -p /root/input /root/results

# Script de inicialização
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 11434

CMD ["/start.sh"]
