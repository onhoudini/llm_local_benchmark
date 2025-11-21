#!/bin/bash

MODEL=$1
DATASET=~/Documentos/pesquisa/ollama-test/input/ask_bench.csv

RUN_ID=$(date +"%Y%m%d-%H%M%S")
OUTDIR=~/Documentos/pesquisa/ollama-test/results/run_$RUN_ID
mkdir -p "$OUTDIR"

OUTCSV="$OUTDIR/resultados.csv"
echo "run_id,pergunta_id,pergunta,resposta,real_time_sec,cpu_percent,max_ram_kb" > "$OUTCSV"

echo "=== BENCHMARK DATASET ==="
echo "Modelo: $MODEL"
echo "Run ID: $RUN_ID"
echo "-------------------------------"

tail -n +2 "$DATASET" | while IFS=',' read -r ID PERGUNTA; do

    CLEAN_Q=$(echo "$PERGUNTA" | sed 's/^"//; s/"$//')
    LOGFILE="$OUTDIR/pergunta_${ID}.log"

    echo "Rodando pergunta $ID: $CLEAN_Q"

    TMP_TIME=$(mktemp)
    TMP_SPINNER=$(mktemp)
    RESPOSTA=$(mktemp)

    
    (
        /usr/bin/time -f "%e %P %M" \
        bash -c "printf \"%s\" \"$CLEAN_Q\" | ollama run \"$MODEL\"" \
        > "$RESPOSTA"
    ) 2> "$TMP_TIME"  \
      2> "$TMP_SPINNER"

    
    read REAL CPU RAM < "$TMP_TIME"

    RESP_TXT=$(cat "$RESPOSTA" | tr '\n' ' ' | sed 's/,/ /g')

    echo "$RUN_ID,$ID,\"$CLEAN_Q\",\"$RESP_TXT\",$REAL,${CPU//%},$RAM" >> "$OUTCSV"

    {
        echo "Pergunta ID: $ID"
        echo "Pergunta: $CLEAN_Q"
        echo ""
        echo "Resposta:"
        cat "$RESPOSTA"
        echo ""
        echo "Real time: $REAL"
        echo "CPU: $CPU"
        echo "Max RAM: $RAM KB"
    } > "$LOGFILE"

    rm "$TMP_TIME" "$TMP_SPINNER" "$RESPOSTA"

done

echo "=== Benchmark concluído ==="
echo "Resultados: $OUTDIR"
