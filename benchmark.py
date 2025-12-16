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

DATASET_PATH = Path.home() / "Documentos/pesquisa/ollama-test/input/NQ-open.efficientqa.dev.1.1.sample.jsonl"
RESULTS_DIR = Path.home() / "Documentos/pesquisa/ollama-test/results"
OLLAMA_API = "http://localhost:11434/api/generate"

MODELS = ["mistral_v0.1", "mistral_v0.2", "mistral_v0.3"]
NUM_EXECUTIONS = 3


def benchmark_question(model, question):
    start_time = time.time()
    gpu_max = 0
    gpu_mem_max = 0
    ram_max = 0
    
    try:
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
        
        while response.poll() is None:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.load * 100 > gpu_max:
                        gpu_max = gpu.load * 100
                    if gpu.memoryUsed > gpu_mem_max:
                        gpu_mem_max = gpu.memoryUsed
                
                try:
                    proc = psutil.Process(response.pid)
                    ram = proc.memory_info().rss // 1024
                    if ram > ram_max:
                        ram_max = ram
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
            except Exception as e:
                pass
            
            time.sleep(0.1)
        
        stdout, stderr = response.communicate()
        result = json.loads(stdout.decode())
        resposta = result.get("response", "").strip()
        elapsed = time.time() - start_time
        
        return resposta, elapsed, int(gpu_max), int(gpu_mem_max * 1024)
    
    except Exception as e:
        return f"Erro: {str(e)}", time.time() - start_time, 0, 0


def run_benchmark_for_model_execution(model, execution_number):
    model_dir = RESULTS_DIR / model.replace("mistral_", "v")
    out_dir = model_dir / f"exec{execution_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "resultados.csv"
    
    print("=" * 60)
    print(f"MODELO: {model} | EXECUÇÃO: {execution_number}")
    print("=" * 60)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "modelo", "execucao", "pergunta_id", "pergunta", "resposta",
            "real_time_sec", "gpu_percent", "gpu_mem_kb"
        ])
        
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for q_id, line in enumerate(f, 1):
                data = json.loads(line)
                pergunta = data.get("question", "")
                
                print(f"[{model} - exec{execution_number}] Pergunta {q_id}: {pergunta[:50]}...")
                
                resposta, elapsed, gpu, gpu_mem = benchmark_question(model, pergunta)
                resposta_clean = resposta.replace("\n", " ").replace(",", " ")
                
                writer.writerow([
                    model, execution_number, q_id, pergunta, resposta_clean,
                    elapsed, gpu, gpu_mem
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
                    log.write(f"GPU Memory (max): {gpu_mem} KB\n")
    
    print(f"✓ Resultados salvos em: {out_dir}\n")


def main():
    print("\n" + "=" * 70)
    print("INICIANDO BENCHMARK MULTI-MODELO COM MÚLTIPLAS EXECUÇÕES")
    print(f"Modelos: {', '.join(MODELS)}")
    print(f"Execuções por modelo: {NUM_EXECUTIONS}")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    for model in MODELS:
        for execution in range(1, NUM_EXECUTIONS + 1):
            try:
                run_benchmark_for_model_execution(model, execution)
            except Exception as e:
                print(f"✗ Erro ao testar {model} - execução {execution}: {e}\n")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 70)
    print("TODOS OS BENCHMARKS CONCLUÍDOS!")
    print("=" * 70)
    print(f"\nDuração total: {duration}")
    print(f"Resultados em: {RESULTS_DIR}")
    for model in MODELS:
        model_name = model.replace("mistral_", "v")
        print(f"  {model_name}/")
        for i in range(1, NUM_EXECUTIONS + 1):
            print(f"    exec{i}/")


if __name__ == "__main__":
    main()