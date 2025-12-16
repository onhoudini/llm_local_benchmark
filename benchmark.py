#!/usr/bin/env python3

import os
import sys
import csv
import json
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import GPUtil
except ImportError:
    print("Instale GPUtil: pip install gputil")
    sys.exit(1)

# Configuração
DATASET_PATH = Path.home() / "Documentos/pesquisa/ollama-test/input/NQ-open.efficientqa.dev.1.1.sample.jsonl"
RESULTS_DIR = Path.home() / "Documentos/pesquisa/ollama-test/results"
OLLAMA_API = "http://localhost:11434/api/generate"

# Modelos a testar
MODELS = ["mistral_v0.1", "mistral_v0.2", "mistral_v0.3"]

def benchmark_question(model, question):
    """Executa uma pergunta e coleta métricas."""
    start_time = time.time()
    
    gpu_max = 0
    gpu_mem_max = 0
    ram_max = 0
    
    try:
        # Prompt melhorado para respostas curtas
        prompt = f"Q: {question}\nA:"
        
        response = subprocess.Popen(
            ["curl", "-s", "-X", "POST", OLLAMA_API,
             "-H", "Content-Type: application/json",
             "-d", json.dumps({
                 "model": model,
                 "prompt": prompt,
                 "stream": False
             })],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Monitora processo enquanto roda
        while response.poll() is None:
            try:
                # Monitora GPU
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.load * 100 > gpu_max:
                        gpu_max = gpu.load * 100
                    if gpu.memoryUsed > gpu_mem_max:
                        gpu_mem_max = gpu.memoryUsed
                
                # Monitora RAM do sistema
                try:
                    proc = psutil.Process(response.pid)
                    ram = proc.memory_info().rss // 1024  # Em KB
                    if ram > ram_max:
                        ram_max = ram
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
            except Exception as e:
                pass
            
            time.sleep(0.1)
        
        # Coleta resposta
        stdout, stderr = response.communicate()
        result = json.loads(stdout.decode())
        resposta = result.get("response", "").strip()
        
        elapsed = time.time() - start_time
        
        return resposta, elapsed, int(gpu_max), int(gpu_mem_max * 1024)  # Converte MB para KB
    
    except Exception as e:
        return f"Erro: {str(e)}", time.time() - start_time, 0, 0


def run_benchmark_for_model(model, run_number):
    """Executa benchmark completo para um modelo."""
    out_dir = RESULTS_DIR / str(run_number)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "resultados.csv"
    
    print("=" * 60)
    print(f"BENCHMARK {run_number} - MODELO: {model}")
    print("=" * 60)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "modelo", "pergunta_id", "pergunta", "resposta",
            "real_time_sec", "gpu_percent", "gpu_mem_kb"
        ])
        
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for q_id, line in enumerate(f, 1):
                data = json.loads(line)
                pergunta = data.get("question", "")
                
                print(f"[{model}] Pergunta {q_id}: {pergunta[:50]}...")
                
                resposta, elapsed, gpu, gpu_mem = benchmark_question(model, pergunta)
                
                # Remove quebras de linha e commas da resposta
                resposta_clean = resposta.replace("\n", " ").replace(",", " ")
                
                writer.writerow([
                    model, q_id, pergunta, resposta_clean,
                    elapsed, gpu, gpu_mem
                ])
                
                # Salva log individual
                log_path = out_dir / f"pergunta_{q_id}.log"
                with open(log_path, "w", encoding="utf-8") as log:
                    log.write(f"Modelo: {model}\n")
                    log.write(f"Pergunta ID: {q_id}\n")
                    log.write(f"Pergunta: {pergunta}\n")
                    log.write("\n")
                    log.write("Resposta:\n")
                    log.write(resposta + "\n")
                    log.write("\n")
                    log.write(f"Real time: {elapsed:.2f}s\n")
                    log.write(f"GPU Load (max): {gpu:.1f}%\n")
                    log.write(f"GPU Memory (max): {gpu_mem} KB\n")
    
    print(f"✓ Resultados salvos em: {out_dir}\n")


def main():
    print("\n" + "=" * 60)
    print("INICIANDO BENCHMARK MULTI-MODELO")
    print("=" * 60 + "\n")
    
    for run_number, model in enumerate(MODELS, 1):
        try:
            run_benchmark_for_model(model, run_number)
        except Exception as e:
            print(f"✗ Erro ao testar modelo {model}: {e}\n")
    
    print("=" * 60)
    print("TODOS OS BENCHMARKS CONCLUÍDOS!")
    print("=" * 60)
    print(f"\nResultados em: {RESULTS_DIR}")
    for i in range(1, len(MODELS) + 1):
        print(f"  {i}/ {RESULTS_DIR / str(i)}")


if __name__ == "__main__":
    main()