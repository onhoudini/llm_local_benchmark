import json
import csv
import statistics
from pathlib import Path

DATASET_PATH = Path.home() / "Documentos/pesquisa/ollama-test/input/NQ-open.efficientqa.dev.1.1.sample.jsonl"
RESULTS_DIR = Path.home() / "Documentos/pesquisa/ollama-test/results"


def load_expected_answers():
    answers = {}
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            data = json.loads(line)
            answers[line_num] = [a.lower() for a in data.get("answer", [])]
    return answers


def check_answer_match(generated, expected_list):
    if not generated or not expected_list:
        return False
    
    generated_lower = generated.lower()
    
    for expected in expected_list:
        if expected in generated_lower:
            return True
    
    return False


def analyze_model_execution(model_path):
    csv_path = model_path / "resultados.csv"
    
    if not csv_path.exists():
        return None
    
    expected_answers = load_expected_answers()
    results = []
    metrics = {
        "time": [],
        "gpu_percent": [],
        "gpu_mem_kb": []
    }
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = int(row["pergunta_id"])
            resposta = row["resposta"].strip()
            real_time = float(row["real_time_sec"])
            gpu = float(row["gpu_percent"])
            mem = float(row["gpu_mem_kb"])
            
            expected = expected_answers.get(q_id, [])
            is_match = check_answer_match(resposta, expected)
            
            results.append({
                "q_id": q_id,
                "pergunta": row["pergunta"],
                "resposta": resposta,
                "esperada": " | ".join(expected),
                "match": is_match,
                "time": real_time,
                "gpu": gpu,
                "mem": mem
            })
            
            metrics["time"].append(real_time)
            metrics["gpu_percent"].append(gpu)
            metrics["gpu_mem_kb"].append(mem)
    
    return results, metrics


def generate_report(model_name, results, metrics):
    if not results:
        return
    
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"ANÁLISE: {model_name}")
    report.append(f"{'='*80}\n")
    
    # Factualidade
    matches = sum(1 for r in results if r["match"])
    accuracy = (matches / len(results)) * 100
    
    report.append(f"FACTUALIDADE")
    report.append(f"-" * 40)
    report.append(f"Acurácia: {accuracy:.2f}% ({matches}/{len(results)})")
    report.append("")
    
    # Tempo
    report.append(f"DESEMPENHO - TEMPO")
    report.append(f"-" * 40)
    report.append(f"Média: {statistics.mean(metrics['time']):.2f}s")
    report.append(f"Mediana: {statistics.median(metrics['time']):.2f}s")
    report.append(f"Min/Max: {min(metrics['time']):.2f}s / {max(metrics['time']):.2f}s")
    if len(metrics['time']) > 1:
        report.append(f"Desvio padrão: {statistics.stdev(metrics['time']):.2f}s")
    report.append("")
    
    # GPU
    report.append(f"DESEMPENHO - GPU")
    report.append(f"-" * 40)
    report.append(f"Carga média: {statistics.mean(metrics['gpu_percent']):.1f}%")
    report.append(f"Carga mediana: {statistics.median(metrics['gpu_percent']):.1f}%")
    report.append(f"Carga min/max: {min(metrics['gpu_percent']):.1f}% / {max(metrics['gpu_percent']):.1f}%")
    if len(metrics['gpu_percent']) > 1:
        report.append(f"Desvio padrão: {statistics.stdev(metrics['gpu_percent']):.1f}%")
    report.append("")
    
    # VRAM
    report.append(f"DESEMPENHO - MEMÓRIA")
    report.append(f"-" * 40)
    mem_mb = [m / 1024 for m in metrics['gpu_mem_kb']]
    report.append(f"Média: {statistics.mean(mem_mb):.1f} MB")
    report.append(f"Mediana: {statistics.median(mem_mb):.1f} MB")
    report.append(f"Min/Max: {min(mem_mb):.1f} MB / {max(mem_mb):.1f} MB")
    if len(mem_mb) > 1:
        report.append(f"Desvio padrão: {statistics.stdev(mem_mb):.1f} MB")
    report.append("")
    
    return "\n".join(report)


def main():
    print("Analisando resultados de benchmark...\n")
    
    all_reports = []
    
    for model_dir in sorted(RESULTS_DIR.glob("v*")):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        for exec_dir in sorted(model_dir.glob("exec*")):
            results, metrics = analyze_model_execution(exec_dir) or (None, None)
            
            if results:
                report = generate_report(f"{model_name} - {exec_dir.name}", results, metrics)
                all_reports.append(report)
                print(report)
    
    # Relatório
    report_path = RESULTS_DIR / "ANALYSIS_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_reports))
    
    print(f"\n{'='*80}")
    print(f"Relatório salvo em: {report_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()