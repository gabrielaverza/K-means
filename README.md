# K-means 1D — Versão Sequencial, OpenMP e CUDA

Este projeto implementa o algoritmo **K-means 1D** em diferentes versões:

* **Naive (sequencial – baseline)**
* **OpenMP (paralelização em CPU)**
* **CUDA (paralelização em GPU)**

O objetivo é **avaliar o impacto da paralelização** no tempo de execução, throughput, custo de comunicação e speedup, produzindo métricas numéricas e gráficos automáticos.

---

## 📁 Estrutura do Repositório

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
├── run_bench.sh               # Script de benchmark unificado
├── analisar_bench.py          # Consolidação + gráficos + validação
├── figs_bench/                # Gerado automaticamente
│   ├── openmp/
│   ├── cuda/
│   └── global/    
└── README.md
```

---

# 📌 Descrição das Versões e das Métricas

## 🔹 **1. Naive (Sequencial – CPU)**

Versão baseline usada como referência para speedup.

**Métricas extraídas:**

* Tempo total de execução
* Iterações até convergência
* SSE final (para verificação de corretude)

---

## 🔹 **2. OpenMP (CPU paralela)**

Utiliza paralelização com múltiplas threads e diferentes configurações:

* Threads: 1, 2, 4, 8, 16
* Schedules: `static` e `dynamic`
* Chunk sizes: 1, 64, 256, 1024

**Métricas extraídas:**

* Tempo de execução (mediana de 5 execuções)
* Speedup em relação ao sequencial
* Throughput (pontos/s)
* SSE final
* Comparação entre escalonamentos e chunks

---

## 🔹 **3. CUDA (GPU)**

Implementação paralela utilizando kernels CUDA.

**Métricas extraídas:**

* Tempo de cópia Host → Device (H2D)
* Tempo de cópia Device → Host (D2H)
* Tempo de execução do kernel
* Tempo total da execução
* Throughput (pontos/s)
* Speedup vs. sequencial e vs. OpenMP
* Tamanhos de:

  * **grid**
  * **block** (ex.: 128, 256, 512)

Tudo isso já é coletado automaticamente pelo `run_bench.sh`.

---

# 🚀 Como Executar

## 1️⃣ Gerar conjuntos de teste

```bash
cd conjuntos_teste
chmod +x pipeline.sh
./pipeline.sh
```

Isso cria automaticamente os conjuntos **p**, **m** e **g**.

---

## 2️⃣ Executar benchmark

O script `run_bench.sh` aceita parâmetros:

### 🔸 Rodar **apenas sequencial + OpenMP**

```
./run_bench.sh --omp
```

### 🔸 Rodar **sequencial + CUDA**

```
./run_bench.sh --cuda
```

### 🔸 Rodar **somente sequencial**

```
./run_bench.sh
```

### 🔸 Rodar **todas as versões**

```
./run_bench.sh --omp --cuda
```

### 📌 O que o script faz automaticamente:

* Compila Naive, OpenMP e/ou CUDA conforme parâmetros
* Roda benchmarks completos com 5 repetições
* Gera medições, medianas e speedups
* Cria nomes de CSV como:

```
resultados_omp_YYYYMMDD_HHMMSS.csv
resultados_cuda_YYYYMMDD_HHMMSS.csv
resultados_omp_cuda_YYYYMMDD_HHMMSS.csv
```

---

## 3️⃣ Gerar gráficos e tabelas com Python

O script `analisar_bench.py` recebe dois parâmetros:

```
python3 analisar_bench.py <arquivo_csv> <modo>
```

### 🔸 Processar **somente resultados CUDA**

```
python3 analisar_bench.py resultados_cuda.csv cuda
```

### 🔸 Processar **somente resultados OpenMP**

```
python3 analisar_bench.py resultados_omp.csv omp
```

### 🔸 Processar **todas as versões juntas (Serial + OpenMP + CUDA)**

```
python3 analisar_bench.py resultados_omp_cuda.csv all
```

O script identifica automaticamente os modos presentes (Serial, OpenMP, CUDA) e gera a seguinte estrutura:

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
└── global/
    └── comparacao_seq_omp_cuda.csv
```

A pasta **openmp/** contém gráficos de:

* tempo × threads
* throughput × threads
* speedup vs. sequência
* efeitos de scheduler e chunk

A pasta **cuda/** contém gráficos de:

* tempo × block size
* throughput × block size
* speedup vs. serial e vs. OpenMP

A pasta **global/** contém:

* **`comparacao_seq_omp_cuda.csv`** — tabela consolidada comparando Serial × OpenMP × CUDA
  (usada para gerar tabelas de avaliação no relatório)

Além disso, o script também gera:

* **`validacao_sse.txt`** — confirma corretude entre todas as versões
* Relatório no terminal com as melhores configurações encontradas por modo

---

# 🧰 Dependências

### **Para compilação**

* GCC com suporte a OpenMP
* NVCC (CUDA Toolkit)

### **Para análise**

```
source .venv/bin/activate
pip install -r requirements.txt
```

### Ambiente recomendado

* **WSL2 + VSCode**
* GPU NVIDIA com CUDA disponível (para testes CUDA)