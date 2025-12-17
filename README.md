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

1. Instale Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Baixe o modelo na versão e quantização adequada

## Dependências Atuais
- Ollama