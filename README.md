# LLM Local Benchmark Test

Projeto para benchmarking de modelos no Ollama usando um dataset de 100 perguntas.

## Estrutura Atual

### Códigos

- `benchmark.py`: Benchmark dos modelos.
- `analyze.py`: Análise dos resultados do benchmark.

### Pastas

- `input/`: Dataset JSONL com perguntas.
- `models/`: Arquivo contendo arquivos Modelfile para configuração dos modelos.
- `results/`: Resultados gerados pelo benchmark.py e analyze.py.


## Setup

1. Baixe e instale o Docker
2. Execute o Dockerfile com o comando
```
docker build -t ollama-benchmark .
```
3. Execute o comando run do Docker
```
docker run --rm -e BENCHMARK_INPUT_DIR=/data/input -e BENCHMARK_RESULTS_DIR=/data/results -v "$(pwd)/input:/data/input" -v "$(pwd)/results:/data/results" ollama-benchmark
```

## Dependências Atuais
- Docker
