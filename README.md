# 📘 **K-means - Sequencial, OpenMP, CUDA e MPI**

Este projeto implementa o algoritmo **K-means 1D** em múltiplas arquiteturas de paralelização:

* **Naive (Sequencial – baseline)**
* **OpenMP (CPU multithread)**
* **CUDA (GPU)**
* **MPI (processamento distribuído com múltiplos processos)**

O objetivo é avaliar:

* **Strong scaling**: desempenho à medida que aumentamos P (processos/threads)
* **Custo de comunicação** (ex.: Allreduce no MPI)
* **Throughput (pontos/s)**
* **Speedup**
* **Corretude (comparação de SSE)**

Toda a execução, coleta de dados e análise gráfica é automatizada pelos scripts incluídos.

---

# 📁 **Estrutura do Repositório**

```
.
├── conjuntos_teste/
│   ├── pipeline.sh
│   ├── dados_p.csv
│   ├── dados_m.csv
│   ├── dados_g.csv
│   ├── centroides_iniciais_p.csv
│   ├── centroides_iniciais_m.csv
│   └── centroides_iniciais_g.csv
│
├── serial/
│   └── kmeans_1d_naive.c
│
├── openmp/
│   └── kmeans_1d_omp.c
│
├── cuda/
│   └── kmeans_1d_cuda.cu
│
├── mpi/
│   └── kmeans_1d_mpi.c
│
├── run_bench.sh               # Script unificado de benchmark
├── analisar_bench.py          # Gráficos, tabelas, speedups e validação
├── figs_bench/                # Saída automática dos gráficos
│   ├── openmp/
│   ├── cuda/
│   ├── mpi/
│   └── global/
└── README.md
```

---

# 📌 **Descrição das Versões e Métricas**

## 🔹 **1. Naive (Sequencial – CPU)**

Versão básica usada como baseline.

**Métricas:**

* Tempo total
* Iterações
* SSE final (para verificação de corretude)

---

## 🔹 **2. OpenMP (CPU multithread)**

Configurações testadas:

* Threads: `1, 2, 4, 8, 16`
* Schedules: `static`, `dynamic`
* Chunk sizes: `1, 64, 256, 1024`

**Métricas:**

* Tempo (mediana de 5 execuções)
* Speedup vs. sequencial
* Throughput (pontos/s)
* SSE final
* Comparação de escalonamento e chunk

---

## 🔹 **3. CUDA (GPU)**

Implementação paralela com kernels CUDA.

**Métricas:**

* H2D, Kernel, D2H
* Tempo total
* Throughput
* Speedup vs. sequencial e vs OpenMP
* Grid size e block size

---

## 🔹 **4. MPI (Processos distribuídos)**

Versão paralela com **MPI**, baseada na divisão do vetor de pontos entre os processos.

Cada iteração faz:

1. **Broadcast** dos centróides (C)
2. **Assignment local** em cada processo
3. **Reduções globais**:

   * `MPI_Reduce` para SSE
   * `MPI_Allreduce` para somas e contagens
4. **Update global** dos centróides

**Métricas extraídas:**

* Tempo total
* Tempo de comunicação (Allreduce)
* Tempo de computação aproximado
* Strong scaling para P = 1, 2, 4, 8, …
* Speedup vs. sequencial
* Speedup vs. melhor OpenMP
* Throughput (pontos/s)

---

# 🚀 Como Executar

## 1️⃣ **Gerar conjuntos de teste**

```bash
cd conjuntos_teste
chmod +x pipeline.sh
./pipeline.sh
```

---

# 2️⃣ **Executar os benchmarks**

O script unificado aceita flags:

* `--omp`
* `--cuda`
* `--mpi`

### 🔸 Somente Sequencial + MPI

```bash
./run_bench.sh --mpi
```

### 🔸 Sequencial + OpenMP

```bash
./run_bench.sh --omp
```

### 🔸 Sequencial + CUDA

```bash
./run_bench.sh --cuda
```

### 🔸 Todas as versões (seq + omp + cuda + mpi)

```bash
./run_bench.sh --omp --cuda --mpi
```

### 📌 Saída do script

Gera arquivos no formato (exemplos):

```
resultados_omp_mpi_YYYYMMDD_HHMMSS.csv
resultados_omp_cuda_mpi_YYYYMMDD_HHMMSS.csv
resultados_mpi_YYYYMMDD_HHMMSS.csv
```

Incluindo:

* tempos
* iterações
* SSE final
* tempo de comunicação (MPI)
* throughput
* parâmetros (threads, blocks, processos, schedule)

---

# 3️⃣ **Gerar gráficos e tabelas**

Comandos:

```
python3 analisar_bench.py <arquivo_csv> --mpi
python3 analisar_bench.py <arquivo_csv> --openmp
python3 analisar_bench.py <arquivo_csv> --cuda
python3 analisar_bench.py <arquivo_csv> --all
```

### 🔸 Exemplo: comparar **naive x OpenMP x MPI**

```bash
python3 analisar_bench.py resultados_mpi_YYYYMMDD_HHMMSS.csv --omp --mpi
```

### 🔸 Rodar tudo

```bash
python3 analisar_bench.py resultados_omp_cuda_mpi_YYYYMMDD_HHMMSS.csv --all
```

---

# 📊 Estrutura de Saída dos Gráficos

```
figs_bench/
├── openmp/
│   ├── p_omp_*.png
│   ├── m_omp_*.png
│   └── g_omp_*.png
├── cuda/
│   ├── p_cuda_*.png
│   ├── m_cuda_*.png
│   └── g_cuda_*.png
├── mpi/
│   ├── p_mpi_*.png
│   ├── p_mpi_*.png
│   └── p_mpi_*.png
└── global/
    └── comparacao_seq_omp_cuda_mpi.csv
```

---

# 🧰 Dependências

### Compilação:

* GCC com OpenMP
* NVCC (para CUDA)
* MPI (OpenMPI ou MPICH)

### Ambiente Python:

```
source .venv/bin/activate
pip install -r requirements.txt
```