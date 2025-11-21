MODEL=$1
PROMPT=$2

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUT=~/ollama-tests/results/${MODEL}_${TIMESTAMP}.log

echo "=== BENCHMARK OLLAMA ==="        | tee  $OUT
echo "Modelo: $MODEL"                 | tee -a $OUT
echo "Prompt: $PROMPT"                | tee -a $OUT
echo "Data: $(date)"                  | tee -a $OUT
echo "------------------------------" | tee -a $OUT

/usr/bin/time -f "\nReal time: %E\nCPU usage: %P\nMax RAM: %M KB\n" \
    ollama run "$MODEL" "$PROMPT" 2>&1  | tee -a $OUT
