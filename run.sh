#!/bin/bash

# Marca o tempo inicial
start_time=$(date +%s)

# Executa o primeiro script
python3 process.py

# Executa o segundo script
python3 train.py

# Marca o tempo final
end_time=$(date +%s)

# Calcula o tempo total
total_time=$((end_time - start_time))

# Converte para formato h:mm:ss
hours=$((total_time / 3600))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$((total_time % 60))

# Salva no arquivo
{
  echo "Tempo total (segundos): $total_time"
  printf "Tempo total formatado: %d:%02d:%02d\n" $hours $minutes $seconds
} > tempo_total.txt