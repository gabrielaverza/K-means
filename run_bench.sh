#!/usr/bin/env bash
set -euo pipefail

# ===================== AJUDA =====================
usage() {
  cat <<EOF
Uso: $0 [opções]

Por padrão roda apenas o sequencial (naive).
Acrescente --omp, --cuda e/ou --mpi para rodar também essas variantes.

Opções:
  --omp                    Executa seção OpenMP
  --cuda                   Executa seção CUDA
  --mpi                    Executa seção MPI
  --reps N                 Número de repetições por combinação (default: 5)
  --threads "1,2,4,8,16"   Lista de threads p/ OMP (default: "1,2,4,8,16")
  --procs "1,2,4,8"        Lista de processos p/ MPI (default: "1,2,4,8")
  --blocksizes "128,256"   Lista de block sizes p/ CUDA (default: "128,256,512")
  --datasets "p,m,g"       Quais datasets (default: "p,m,g")
  --max-iter N             Máximo de iterações (default: 50)
  --eps VAL                Tolerância (default: 1e-6)
  -h | --help              Mostra esta ajuda

Exemplos:
  # naive + OMP
  $0 --omp

  # naive + CUDA com blocks 256 e 512
  $0 --cuda --blocksizes "256,512"

  # naive + MPI (strong scaling em P=1,2,4,8)
  $0 --mpi --procs "1,2,4,8"

  # naive + OMP + CUDA + MPI
  $0 --omp --cuda --mpi --reps 3 --threads "1,4,8" --procs "1,2,4" --blocksizes "128,256"
EOF
}

# ===================== PARÂMETROS =====================
DO_OMP=0
DO_CUDA=0
DO_MPI=0
REPS=5
THREADS_STR="1,2,4,8,16"
PROCS_STR="1,2,4,8"
BLOCKSIZES_STR="128,256,512"
DATASETS_STR="p,m,g"
MAX_ITER=50
EPS="1e-6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omp) DO_OMP=1; shift ;;
    --cuda) DO_CUDA=1; shift ;;
    --mpi) DO_MPI=1; shift ;;
    --reps) REPS="${2:?}"; shift 2 ;;
    --threads) THREADS_STR="${2:?}"; shift 2 ;;
    --procs) PROCS_STR="${2:?}"; shift 2 ;;
    --blocksizes) BLOCKSIZES_STR="${2:?}"; shift 2 ;;
    --datasets) DATASETS_STR="${2:?}"; shift 2 ;;
    --max-iter) MAX_ITER="${2:?}"; shift 2 ;;
    --eps) EPS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opção desconhecida: $1"; usage; exit 1 ;;
  esac
done

# Converte listas string -> arrays bash
IFS=',' read -r -a THREADS <<< "$THREADS_STR"
IFS=',' read -r -a PROCS   <<< "$PROCS_STR"
IFS=',' read -r -a BLOCKSIZES <<< "$BLOCKSIZES_STR"
IFS=',' read -r -a DATASETS <<< "$DATASETS_STR"

# ===================== COMPILAÇÃO (CPU / GPU / MPI) =====================
SERIAL_DIR="./serial"
OMP_DIR="./openmp"
CUDA_DIR="./cuda"
MPI_DIR="./mpi"

# --- Compilação do sequencial (sempre) ---
echo "==> Compilando kmeans_1d_naive.c (sem OpenMP, usa clock())"
gcc -O2 -std=c99 "${SERIAL_DIR}/kmeans_1d_naive.c" -o kmeans_1d_naive -lm
[ -x ./kmeans_1d_naive ] || { echo "ERRO: kmeans_1d_naive não gerado."; exit 1; }

# --- Compilação do OpenMP (somente se --omp foi passado) ---
if (( DO_OMP )); then
  echo "==> Compilando kmeans_1d_omp.c (com OpenMP)"
  if [[ -f "${OMP_DIR}/kmeans_1d_omp.c" ]]; then
    gcc -O2 -fopenmp -std=c99 "${OMP_DIR}/kmeans_1d_omp.c" -o kmeans_1d_omp -lm
    [ -x ./kmeans_1d_omp ] || { echo "ERRO: kmeans_1d_omp não gerado."; exit 1; }
  else
    echo "ERRO: arquivo ${OMP_DIR}/kmeans_1d_omp.c não encontrado!"
    exit 1
  fi
fi

# --- Compilação do CUDA (somente se --cuda foi passado) ---
if (( DO_CUDA )); then
  if command -v nvcc >/dev/null 2>&1; then
    if [[ -f "${CUDA_DIR}/kmeans_1d_cuda.cu" ]]; then
      echo "==> Compilando kmeans_1d_cuda.cu (CUDA)"
      nvcc -O2 "${CUDA_DIR}/kmeans_1d_cuda.cu" -o kmeans_1d_cuda
      [ -x ./kmeans_1d_cuda ] || { echo "ERRO: kmeans_1d_cuda não gerado."; exit 1; }
    else
      echo "ERRO: arquivo ${CUDA_DIR}/kmeans_1d_cuda.cu não encontrado!"
      exit 1
    fi
  else
    echo "ERRO: compilador nvcc não encontrado no PATH."
    echo "→ Dica: instale o CUDA Toolkit e adicione o caminho do nvcc ao PATH."
    exit 1
  fi
fi

# --- Compilação do MPI (somente se --mpi foi passado) ---
if (( DO_MPI )); then
  echo "==> Compilando kmeans_1d_mpi.c (MPI)"
  if command -v mpicc >/dev/null 2>&1; then
    if [[ -f "${MPI_DIR}/kmeans_1d_mpi.c" ]]; then
      mpicc -O2 -std=c99 "${MPI_DIR}/kmeans_1d_mpi.c" -o kmeans_1d_mpi -lm
      [ -x ./kmeans_1d_mpi ] || { echo "ERRO: kmeans_1d_mpi não gerado."; exit 1; }
    else
      echo "ERRO: arquivo ${MPI_DIR}/kmeans_1d_mpi.c não encontrado!"
      exit 1
    fi
  else
    echo "ERRO: mpicc não encontrado no PATH."
    echo "→ Dica: instale OpenMPI ou MPICH e adicione mpicc/mpirun ao PATH."
    exit 1
  fi
fi

# ===================== CONFIGURAÇÃO DE ARQUIVOS =====================
BASE_DIR="./conjuntos_teste"

declare -A DADOS_MAP CENTR_MAP
DADOS_MAP=( 
  ["p"]="${BASE_DIR}/dados_p.csv"
  ["m"]="${BASE_DIR}/dados_m.csv"
  ["g"]="${BASE_DIR}/dados_g.csv"
)
CENTR_MAP=(
  ["p"]="${BASE_DIR}/centroides_iniciais_p.csv"
  ["m"]="${BASE_DIR}/centroides_iniciais_m.csv"
  ["g"]="${BASE_DIR}/centroides_iniciais_g.csv"
)

SCHEDULES_BASE=("static" "dynamic")
SCHEDULES_TUNE=("static" "dynamic")
CHUNKS=(1 64 256 1024)

STAMP=$(date +"%Y%m%d_%H%M%S")

if (( DO_OMP )) && (( DO_CUDA )) && (( DO_MPI )); then
    MODE="omp_cuda_mpi"
elif (( DO_OMP )) && (( DO_CUDA )); then
    MODE="omp_cuda"
elif (( DO_OMP )) && (( DO_MPI )); then
    MODE="omp_mpi"
elif (( DO_CUDA )) && (( DO_MPI )); then
    MODE="cuda_mpi"
elif (( DO_MPI )); then
    MODE="mpi"
elif (( DO_CUDA )); then
    MODE="cuda"
elif (( DO_OMP )); then
    MODE="omp"
else
    MODE="seq"
fi

OUTCSV="resultados_${MODE}_${STAMP}.csv"

# ===================== FUNÇÕES AUXILIARES =====================
parse_line_seqomp() {
  awk '
    /Itera/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Itera/)        { iters=$(i+1) }
        if ($i ~ /^SSE/)          { sse=$(i+2) }
        if ($i ~ /^Tempo:/)       { t=$(i+1) }
      }
      gsub(/\r/,"",iters); gsub(/\r/,"",sse); gsub(/\r/,"",t);
      print iters, sse, t
      exit
    }
  '
}

parse_line_cuda() {
  awk '
    /Itera/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Itera/)       { iters=$(i+1) }
        if ($i ~ /^SSE/)         { sse=$(i+2) }
        if ($i ~ /^total:/)      { total=$(i+1) }
      }
      gsub(/\r/,"",iters); gsub(/\r/,"",sse); gsub(/\r/,"",total);
      got1=1
    }
    /Tempos:/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^H2D=/)    { sub(/H2D=/,"",$i); h2d=$i }
        if ($i ~ /^Kernel=/) { sub(/Kernel=/,"",$i); kernel=$i }
        if ($i ~ /^D2H=/)    { sub(/D2H=/,"",$i); d2h=$i }
      }
      gsub(/ms/,"",h2d); gsub(/ms/,"",kernel); gsub(/ms/,"",d2h);
      gsub(/\r/,"",h2d); gsub(/\r/,"",kernel); gsub(/\r/,"",d2h);
      got2=1
    }
    END{
      if (got1) {
        gsub(/ms/,"",total)
        print iters, sse, total, h2d, kernel, d2h
      }
    }
  '
}

# Parse da saída do MPI:
# Espera linhas do tipo:
#   Iterações: X
#   SSE final: Y
#   Tempo total (apenas K-means): Z ms
#   Tempo de comunicação (Allreduce): W ms
parse_line_mpi() {
  awk '
    /Itera/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Iterações:/) { iters=$(i+1) }
      }
      gsub(/\r/,"",iters)
    }
    /SSE final:/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^SSE/) { sse=$(i+2) }
      }
      gsub(/\r/,"",sse)
    }
    /Tempo total/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Tempo/) { total=$(i+4) }  # "Tempo total (apenas K-means): XXX ms"
      }
      gsub(/ms/,"",total); gsub(/\r/,"",total)
    }
    /Tempo de comunicação/ {
      for (i=1; i<=NF; i++) {
        if ($i ~ /^Tempo/) { comm=$(i+4) }  # "Tempo de comunicação (Allreduce): YYY ms"
      }
      gsub(/ms/,"",comm); gsub(/\r/,"",comm)
    }
    END {
      if (iters == "") iters="NA";
      if (sse == "")   sse="NA";
      if (total == "") total="NA";
      if (comm == "")  comm="NA";
      # imprime: iters sse total_ms comm_ms
      print iters, sse, total, comm
    }
  '
}

median() {
  awk '{a[NR]=$1} END{
    if (NR==0) {print "NA"; exit}
    n=asort(a)
    if (n%2==1) print a[(n+1)/2];
    else print (a[n/2]+a[n/2+1])/2.0
  }'
}

run_one_seq() {
  local dados="$1" centr="$2"
  ./kmeans_1d_naive "$dados" "$centr" "$MAX_ITER" "$EPS" | parse_line_seqomp
}
run_one_omp() {
  local dados="$1" centr="$2"
  ./kmeans_1d_omp "$dados" "$centr" "$MAX_ITER" "$EPS" | parse_line_seqomp
}
run_one_cuda() {
  local dados="$1" centr="$2" block="$3"
  ./kmeans_1d_cuda "$dados" "$centr" "$MAX_ITER" "$EPS" "$block" | parse_line_cuda
}
run_one_mpi() {
  local dados="$1" centr="$2" procs="$3"
  mpirun -np "$procs" ./kmeans_1d_mpi "$dados" "$centr" "$MAX_ITER" "$EPS" | parse_line_mpi
}

write_header() {
  echo "dataset,modo,threads,schedule,chunk,block,rep,iters,sse_final,time_ms,median_ms,h2d_ms,kernel_ms,d2h_ms,total_ms" > "$OUTCSV"
}

append_csv_seqomp() {
  local dataset="$1" modo="$2" threads="$3" schedule="$4" chunk="$5" rep="$6"
  local iters="$7" sse="$8" time_ms="$9" median_ms="${10}"
  echo "${dataset},${modo},${threads},${schedule},${chunk},NA,${rep},${iters},${sse},${time_ms},${median_ms},NA,NA,NA,NA" >> "$OUTCSV"
}

append_csv_cuda() {
  local dataset="$1" rep="$2" block="$3"
  local iters="$4" sse="$5" total_ms="$6" h2d_ms="$7" kernel_ms="$8" d2h_ms="$9"
  echo "${dataset},cuda,NA,NA,NA,${block},${rep},${iters},${sse},NA,NA,${h2d_ms},${kernel_ms},${d2h_ms},${total_ms}" >> "$OUTCSV"
}

append_csv_cuda_median() {
  local dataset="$1" block="$2" median_total="$3"
  echo "${dataset},cuda,NA,NA,NA,${block},0,NA,NA,NA,${median_total},NA,NA,NA,${median_total}" >> "$OUTCSV"
}

# Para MPI, usamos:
#   modo=mpi
#   threads = número de processos (P)
#   time_ms = tempo total do K-means
#   h2d_ms  = tempo de comunicação (Allreduce)
#   kernel_ms = tempo de computação aproximado (total - comunicação)
append_csv_mpi() {
  local dataset="$1" procs="$2" rep="$3"
  local iters="$4" sse="$5" total_ms="$6" comm_ms="$7"
  local comp_ms="NA"
  if [[ "$total_ms" != "NA" && "$comm_ms" != "NA" ]]; then
    comp_ms=$(awk -v t="$total_ms" -v c="$comm_ms" 'BEGIN{printf "%.3f", t-c}')
  fi
  echo "${dataset},mpi,${procs},NA,NA,NA,${rep},${iters},${sse},${total_ms},NA,${comm_ms},${comp_ms},NA,${total_ms}" >> "$OUTCSV"
}

append_csv_mpi_median() {
  local dataset="$1" procs="$2" median_total="$3" median_comm="$4"
  local median_comp="NA"
  if [[ "$median_total" != "NA" && "$median_comm" != "NA" ]]; then
    median_comp=$(awk -v t="$median_total" -v c="$median_comm" 'BEGIN{printf "%.3f", t-c}')
  fi
  echo "${dataset},mpi,${procs},NA,NA,NA,0,NA,NA,NA,${median_total},${median_comm},${median_comp},NA,${median_total}" >> "$OUTCSV"
}

# ===================== EXECUÇÃO =====================
write_header

# 1) BASELINE SEQUENCIAL (sempre)
declare -A TEMPO_SERIAL_MS
for ds in "${DATASETS[@]}"; do
  dados="${DADOS_MAP[$ds]}"
  centr="${CENTR_MAP[$ds]}"
  [ -f "$dados" ] || { echo "ERRO: arquivo $dados não existe (dataset ${ds})."; exit 1; }
  [ -f "$centr" ] || { echo "ERRO: arquivo $centr não existe (dataset ${ds})."; exit 1; }

  echo "==> Sequencial: dataset=${ds}"
  times=()
  for rep in $(seq 1 "$REPS"); do
    read iters sse tms < <( run_one_seq "$dados" "$centr" )
    echo "   [seq ${ds}] rep=${rep}  iters=${iters}  SSE=${sse}  time_ms=${tms}"
    append_csv_seqomp "$ds" "seq" 1 "NA" "NA" "$rep" "$iters" "$sse" "$tms" "NA"
    times+=("$tms")
  done
  med=$(printf "%s\n" "${times[@]}" | median)
  TEMPO_SERIAL_MS["$ds"]="$med"
  echo "   [seq ${ds}] mediana(ms)=${med}"
  append_csv_seqomp "$ds" "seq" 1 "NA" "NA" 0 "NA" "NA" "NA" "$med"
done

# 2) OPENMP (opcional)
if (( DO_OMP )); then
  # 2.1) Schedules base (sem chunk)
  for ds in "${DATASETS[@]}"; do
    dados="${DADOS_MAP[$ds]}"; centr="${CENTR_MAP[$ds]}"
    for sched in "${SCHEDULES_BASE[@]}"; do
      export OMP_SCHEDULE="$sched"
      echo "==> OMP: dataset=${ds} schedule=${sched} (sem chunk) — escalonamento em threads"
      for T in "${THREADS[@]}"; do
        export OMP_NUM_THREADS="$T"
        times=()
        for rep in $(seq 1 "$REPS"); do
          read iters sse tms < <( run_one_omp "$dados" "$centr" )
          echo "   [omp ${ds}] T=${T} rep=${rep} iters=${iters} SSE=${sse} time_ms=${tms}"
          append_csv_seqomp "$ds" "omp" "$T" "$sched" "NA" "$rep" "$iters" "$sse" "$tms" "NA"
          times+=("$tms")
        done
        med=$(printf "%s\n" "${times[@]}" | median)
        append_csv_seqomp "$ds" "omp" "$T" "$sched" "NA" 0 "NA" "NA" "NA" "$med"
        echo "   [omp ${ds}] T=${T} schedule=${sched} mediana(ms)=${med}  (baseline seq ms=${TEMPO_SERIAL_MS[$ds]})"
      done
    done
  done

  # 2.2) Afinar chunk (para cada schedule)
  for ds in "${DATASETS[@]}"; do
    dados="${DADOS_MAP[$ds]}"; centr="${CENTR_MAP[$ds]}"
    for sched in "${SCHEDULES_TUNE[@]}"; do
      for chunk in "${CHUNKS[@]}"; do
        export OMP_SCHEDULE="${sched},${chunk}"
        echo "==> OMP: dataset=${ds} schedule=${sched}, chunk=${chunk} — escalonamento em threads"
        for T in "${THREADS[@]}"; do
          export OMP_NUM_THREADS="$T"
          times=()
          for rep in $(seq 1 "$REPS"); do
            read iters sse tms < <( run_one_omp "$dados" "$centr" )
            echo "   [omp ${ds}] T=${T} rep=${rep} iters=${iters} SSE=${sse} time_ms=${tms}"
            append_csv_seqomp "$ds" "omp" "$T" "$sched" "$chunk" "$rep" "$iters" "$sse" "$tms" "NA"
            times+=("$tms")
          done
          med=$(printf "%s\n" "${times[@]}" | median)
          append_csv_seqomp "$ds" "omp" "$T" "$sched" "$chunk" 0 "NA" "NA" "NA" "$med"
          echo "   [omp ${ds}] T=${T} schedule=${sched},chunk=${chunk} mediana(ms)=${med}"
        done
      done
    done
  done
fi

# 3) CUDA (opcional)
if (( DO_CUDA )); then
  for ds in "${DATASETS[@]}"; do
    dados="${DADOS_MAP[$ds]}"; centr="${CENTR_MAP[$ds]}"
    for block in "${BLOCKSIZES[@]}"; do
      echo "==> CUDA: dataset=${ds} block=${block}"
      totals=()
      for rep in $(seq 1 "$REPS"); do
        read iters sse total_ms h2d_ms kernel_ms d2h_ms < <( run_one_cuda "$dados" "$centr" "$block" )
        echo "   [cuda ${ds}] block=${block} rep=${rep} iters=${iters} SSE=${sse} total_ms=${total_ms} (H2D=${h2d_ms} | Kernel=${kernel_ms} | D2H=${d2h_ms})"
        append_csv_cuda "$ds" "$rep" "$block" "$iters" "$sse" "$total_ms" "$h2d_ms" "$kernel_ms" "$d2h_ms"
        totals+=("$total_ms")
      done
      med_total=$(printf "%s\n" "${totals[@]}" | median)
    done
  done
fi

# 4) MPI (opcional)
if (( DO_MPI )); then
  for ds in "${DATASETS[@]}"; do
    dados="${DADOS_MAP[$ds]}"; centr="${CENTR_MAP[$ds]}"
    echo "==> MPI: dataset=${ds} (strong scaling em P=${PROCS_STR})"
    for P in "${PROCS[@]}"; do
      echo "   [mpi ${ds}] P=${P}"
      totals=()
      comms=()
      for rep in $(seq 1 "$REPS"); do
        read iters sse total_ms comm_ms < <( run_one_mpi "$dados" "$centr" "$P" )
        echo "      rep=${rep} iters=${iters} SSE=${sse} total_ms=${total_ms} comm_ms=${comm_ms}"
        append_csv_mpi "$ds" "$P" "$rep" "$iters" "$sse" "$total_ms" "$comm_ms"
        totals+=("$total_ms")
        comms+=("$comm_ms")
      done
      med_total=$(printf "%s\n" "${totals[@]}" | median)
      med_comm=$(printf "%s\n" "${comms[@]}" | median)
      append_csv_mpi_median "$ds" "$P" "$med_total" "$med_comm"
      echo "   [mpi ${ds}] P=${P} mediana total(ms)=${med_total}  mediana comm(ms)=${med_comm}  (baseline seq ms=${TEMPO_SERIAL_MS[$ds]})"
    done
  done
fi

echo
echo "==> Concluído."
echo "CSV consolidado: ${OUTCSV}"
echo "Colunas:"
echo "dataset,modo,threads,schedule,chunk,block,rep,iters,sse_final,time_ms,median_ms,h2d_ms,kernel_ms,d2h_ms,total_ms"