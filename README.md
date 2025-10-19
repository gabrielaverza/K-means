# K-means 1D com OpenMP – Etapa 1 (Paralelização em CPU)

Este projeto implementa e avalia a paralelização do algoritmo **K-means 1D** utilizando **OpenMP**. Foram comparadas as versões sequencial e paralela com diferentes números de threads, políticas de escalonamento (*static* e *dynamic*) e tamanhos de *chunk*, com foco em desempenho (tempo, speedup, throughput) e corretude (SSE e resultados finais dos clusters).

---

## 📁 Estrutura do Repositório

```
.
├── conjuntos_teste/
│   ├── pipeline.sh              # Script para gerar os arquivos de entrada
│   ├── dados_p.csv, centroides_iniciais_p.csv
│   ├── dados_m.csv, centroides_iniciais_m.csv
│   └── dados_g.csv, centroides_iniciais_g.csv
│
├── serial/
│   └── kmeans_1d_naive.c        # Implementação sequencial (baseline)
│
├── openmp/
│   └── kmeans_1d_omp.c          # Versão paralela com OpenMP
│
├── run_bench.sh                 # Script para compilar, executar e medir desempenho
├── analisar_bench.py            # Gera gráficos e valida SSE
├── resultados_YYYYMMDD_HHMMSS.csv  # Arquivo de resultados gerado automaticamente
└── README.md
```

> 🔴 **Importante:** Mesmo estando organizados em pastas, os scripts `run_bench.sh` e `analisar_bench.py` assumem que os arquivos `.c` e os `.csv` estão acessíveis no diretório atual. Para executar, copie ou mova os arquivos ou ajuste os caminhos conforme necessário.

---

## 🚀 Como Executar

### 1️⃣ Gerar os conjuntos de teste

Na pasta `conjuntos_teste/`:

```bash
chmod +x pipeline.sh
./pipeline.sh
```

Isso cria os arquivos de entrada para os três cenários:

* Pequeno (10⁴ pontos, K=4)
* Médio (10⁵ pontos, K=8)
* Grande (10⁶ pontos, K=16)

### 2️⃣ Executar os testes de desempenho

Na raiz do repositório:

```bash
chmod +x run_bench.sh
./run_bench.sh
```

Isso irá:

* Compilar automaticamente as versões sequencial e paralela
* Executar 5 vezes cada configuração de threads, `schedule` e `chunk`
* Gerar um arquivo CSV consolidado

### 3️⃣ Analisar resultados e gerar gráficos

Ainda na raiz:

```bash
source venv/bin/activate  # se estiver usando ambiente virtual
python3 analisar_bench.py resultados_XXXX.csv
```

Serão gerados:

* Gráficos de tempo, speedup e throughput
* Comparação de escalonamento e chunk
* Relatório de validação de SSE (corretude)

---

## 📊 O que o projeto demonstra

* Ganho de desempenho com paralelização em CPU usando OpenMP
* Influência do número de threads no tempo de execução
* Efeito das políticas de escalonamento (`static` vs `dynamic`)
* Importância do tamanho do *chunk* no balanceamento de carga
* Manutenção da corretude (SSE e centróides iguais ao sequencial)

---

## 🔧 Dependências

* **Compilador GCC com suporte a OpenMP**
* **Python 3 + pandas + matplotlib** (para análise e gráficos)
* Ambiente recomendado: **WSL + VS Code**


