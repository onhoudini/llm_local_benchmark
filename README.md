# LLM Local Benchmark Test

Projeto para benchmarking de modelos no Ollama usando um mini dataset de perguntas.

## Estrutura Atual
- `input/`: Dataset CSV com perguntas.
- `models/`: Arquivo Modelfile para configuração do modelo.
- `scripts/`: Scripts Bash para benchmarks.

## Setup
1. Instale Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Baixe o modelo na versão e quantização adequada
3. Execute: `./scripts/bench_dataset.sh <nome_do_modelo>`

## Dependências Atuais
- Bash
- Ollama