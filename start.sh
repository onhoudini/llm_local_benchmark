#!/bin/bash
set -e

echo "🚀 Iniciando Ollama..."
ollama serve &

# Aguarda o Ollama ficar disponível
echo "⏳ Aguardando Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 1
done

echo "✅ Ollama disponível"

# Baixa o modelo (ajuste conforme necessário)
MODEL="mistral:7b"

echo "📥 Baixando modelo: $MODEL"
ollama pull $MODEL

echo "🏁 Iniciando benchmark..."
python3 benchmark.py

echo "📊 Iniciando análise..."
python3 analyze.py

#docker run --rm --gpus all -v "$(pwd)/input:/root/input" -v "$(pwd)/results:/root/results" ollama-benchmark
#docker run --rm -e BENCHMARK_INPUT_DIR=/data/input -e BENCHMARK_RESULTS_DIR=/data/results -v "$(pwd)/input:/data/input" -v "$(pwd)/results:/data/results" ollama-benchmark