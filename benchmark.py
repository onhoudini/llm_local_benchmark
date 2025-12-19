import os
import sys
import csv
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

import psutil

# ===============================
# DETECÇÃO OPCIONAL DE GPU NVIDIA
# ===============================
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# ===============================
# DIRETÓRIOS CONFIGURÁVEIS
# ===============================
BASE_DIR = Path.cwd()

INPUT_DIR = Path(
    os.getenv("BENCHMARK_INPUT_DIR", BASE_DIR / "input")
).resolve()

RESULTS_DIR = Path(
    os.getenv("BENCHMARK_RESULTS_DIR", BASE_DIR / "results")
).resolve()

DATASET_PATH = INPUT_DIR / "QA.jsonl"

OLLAMA_API = "http://localhost:11434/api/generate"

# ===============================
# CONFIGURAÇÕES DE EXECUÇÃO
# ===============================
MODELS = ["mistral"]
NUM_EXECUTIONS = 3


# ===============================
# FUNÇÃO DE BENCHMARK POR PERGUNTA
# ===============================
def benchmark_question(model, question):
    start_time = time.time()

    gpu_load_max = 0
    vram_max = 0

    try:
        prompt = f"Q: {question}\nA:"

        response = subprocess.Popen(
            [
                "curl", "-s", "-X", "POST", OLLAMA_API,
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                })
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        while response.poll() is None:

            if GPU_AVAILABLE:
                try:
                    for gpu in GPUtil.getGPUs():
                        gpu_load_max = max(gpu_load_max, gpu.load * 100)
                        vram_max = max(vram_max, gpu.memoryUsed)
                except Exception:
                    pass

            time.sleep(0.1)

        stdout, stderr = response.communicate()

        result = json.loads(stdout.decode())
        resposta = result.get("response", "").strip()

        elapsed = time.time() - start_time

        vram_kb = int(vram_max * 1024) if GPU_AVAILABLE else 0

        return resposta, elapsed, int(gpu_load_max), vram_kb

    except Exception as e:
        return f"Erro: {str(e)}", time.time() - start_time, 0, 0


# ===============================
# EXECUÇÃO DE UM MODELO
# ===============================
def run_benchmark_for_model_execution(model, execution_number):

    model_dir = RESULTS_DIR / model.replace(":", "_")
    out_dir = model_dir / f"exec{execution_number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "resultados.csv"

    print("=" * 60)
    print(f"MODELO: {model} | EXECUÇÃO: {execution_number}")
    print(f"GPU NVIDIA disponível: {'SIM' if GPU_AVAILABLE else 'NÃO (CPU-only)'}")
    print(f"Resultados em: {out_dir}")
    print("=" * 60)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "modelo",
            "execucao",
            "pergunta_id",
            "pergunta",
            "resposta",
            "real_time_sec",
            "gpu_percent",
            "vram_kb"
        ])

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for q_id, line in enumerate(f, 1):
                data = json.loads(line)
                pergunta = data.get("question", "")

                print(f"[{model} | exec{execution_number}] Pergunta {q_id}: {pergunta[:60]}...")

                resposta, elapsed, gpu, vram = benchmark_question(model, pergunta)
                resposta_clean = resposta.replace("\n", " ").replace(",", " ")

                writer.writerow([
                    model,
                    execution_number,
                    q_id,
                    pergunta,
                    resposta_clean,
                    f"{elapsed:.4f}",
                    gpu,
                    vram
                ])

                log_path = out_dir / f"pergunta_{q_id}.log"
                with open(log_path, "w", encoding="utf-8") as log:
                    log.write(f"Modelo: {model}\n")
                    log.write(f"Execução: {execution_number}\n")
                    log.write(f"Pergunta ID: {q_id}\n")
                    log.write(f"Pergunta: {pergunta}\n\n")
                    log.write("Resposta:\n")
                    log.write(resposta + "\n\n")
                    log.write(f"Real time: {elapsed:.2f}s\n")
                    log.write(f"GPU Load (max): {gpu:.1f}%\n")
                    log.write(f"VRAM (max): {vram} KB\n")

    print(f"✓ Resultados salvos em: {out_dir}\n")


# ===============================
# MAIN
# ===============================
def main():

    print("\n" + "=" * 70)
    print("INICIANDO BENCHMARK MULTI-MODELO")
    print(f"Modelos: {', '.join(MODELS)}")
    print(f"Execuções por modelo: {NUM_EXECUTIONS}")
    print(f"Modo de execução: {'GPU' if GPU_AVAILABLE else 'CPU-only'}")
    print(f"Input: {INPUT_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70 + "\n")

    start_time = datetime.now()

    for model in MODELS:
        for execution in range(1, NUM_EXECUTIONS + 1):
            try:
                run_benchmark_for_model_execution(model, execution)
            except Exception as e:
                print(f"✗ Erro ao testar {model} - execução {execution}: {e}\n")

    duration = datetime.now() - start_time

    print("=" * 70)
    print("TODOS OS BENCHMARKS CONCLUÍDOS!")
    print("=" * 70)
    print(f"Duração total: {duration}")
    print(f"Resultados em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
