#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisa o CSV gerado pelo run_bench.sh e produz gráficos/tabelas comparando:

MODO openmp:
- Comparação de desempenho seq x OpenMP
- Gráficos: Tempo vs Threads, Speedup vs Threads, Pontos/s vs Threads
- Comparação de schedule e chunk
- Validação de SSE entre seq e omp

MODO cuda:
- Comparação de desempenho seq x CUDA
- Gráficos por dataset:
    * Tempos H2D, Kernel, D2H, Total vs block
    * Throughput (pontos/s) vs block
    * Speedup vs block (usando sequencial como baseline)
- Tabela consolidada com métricas CUDA

MODO all:
- Faz tudo de openmp + cuda
- Gera ainda um CSV consolidado comparando:
    * seq, melhor configuração OpenMP e CUDA
    * throughput (pontos/s)
    * speedup vs serial
    * speedup vs melhor OpenMP (para CUDA)

Uso:
    python3 analisar_bench.py resultados_YYYYMMDD_HHMMSS.csv openmp
    python3 analisar_bench.py resultados_YYYYMMDD_HHMMSS.csv cuda
    python3 analisar_bench.py resultados_YYYYMMDD_HHMMSS.csv all

Requisitos:
    pip install -r requirements.txt
"""

import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======== Parâmetros do experimento (ajuste se quiser) ========
# Mapeia dataset -> N (pontos)
DATASET_N = {"p": 10_000, "m": 100_000, "g": 1_000_000}

# T escolhido para gráficos de "efeito de schedule/chunk" em OpenMP
THREADS_FIXED_PREFERRED = 8
TOL_SSE = 1e-9  # tolerância para comparar SSEs


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normaliza strings "NA" / nan em colunas categóricas
    for col in ["schedule", "chunk"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({"nan": "NA"})

    # Normaliza numéricos (se a coluna existir)
    num_cols = [
        "threads", "rep", "iters", "sse_final", "time_ms", "median_ms",
        "block", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms",
        "throughput_pts_s", "speedup_seq", "speedup_omp"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df


def summarize_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai baseline sequencial por dataset (mediana, linha rep=0)."""
    base = (
        df[(df["modo"] == "seq") & (df["rep"] == 0)]
        .groupby("dataset", as_index=False)["median_ms"]
        .min()
        .rename(columns={"median_ms": "tempo_serial_ms"})
    )
    return base


# ===================== FUNÇÕES PARA OPENMP =====================

def pick_threads_fixed_available(df_omp: pd.DataFrame):
    """Escolhe T para gráficos de schedule dinamicamente (prefere 8; senão maior disponível)."""
    candidates = sorted(df_omp["threads"].dropna().unique().tolist())
    if THREADS_FIXED_PREFERRED in candidates:
        return THREADS_FIXED_PREFERRED
    return candidates[-1] if candidates else None


def compute_speedup_openmp(df_med: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona coluna 'speedup' às medianas OpenMP usando a mediana do seq
    do mesmo dataset como referência.
    """
    df = df_med.copy()
    df = df.merge(base, on="dataset", how="left")  # adiciona tempo_serial_ms
    # Para rep=0, usamos median_ms como tempo da config
    df["speedup"] = df["tempo_serial_ms"] / df["median_ms"]
    return df


def plot_time_speedup_throughput_openmp(df_omp_medians: pd.DataFrame, dataset: str, outdir: str):
    """Plota Tempo vs Threads, Speedup vs Threads, Pontos/s vs Threads (usando median_ms)."""
    os.makedirs(outdir, exist_ok=True)

    base_mask = (
        (df_omp_medians["dataset"] == dataset) &
        (df_omp_medians["chunk"] == "NA")
    )
    dsub = df_omp_medians[base_mask].copy()
    if dsub.empty:
        return

    dsub = dsub.sort_values(["schedule", "threads"])

    # Tempo vs Threads
    plt.figure()
    for sched, grp in dsub.groupby("schedule"):
        plt.plot(grp["threads"], grp["median_ms"], marker="o", label=f"{sched}")
    plt.xlabel("Threads")
    plt.ylabel("Tempo (ms)")
    plt.title(f"OpenMP — Tempo vs Threads — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_omp_tempo_vs_threads.png"))
    plt.close()

    # Speedup vs Threads
    plt.figure()
    for sched, grp in dsub.groupby("schedule"):
        plt.plot(grp["threads"], grp["speedup"], marker="o", label=f"{sched}")
    plt.xlabel("Threads")
    plt.ylabel("Speedup (mediana vs seq)")
    plt.title(f"OpenMP — Speedup vs Threads — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_omp_speedup_vs_threads.png"))
    plt.close()

    # Pontos/s vs Threads => N * 1000 / ms
    N = DATASET_N.get(dataset, np.nan)
    if not math.isnan(N):
        dsub = dsub.copy()
        dsub["points_per_s"] = N * 1000.0 / dsub["median_ms"]
        plt.figure()
        for sched, grp in dsub.groupby("schedule"):
            plt.plot(grp["threads"], grp["points_per_s"], marker="o", label=f"{sched}")
        plt.xlabel("Threads")
        plt.ylabel("Pontos por segundo")
        plt.title(f"OpenMP — Pontos/s vs Threads — dataset {dataset}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_omp_pps_vs_threads.png"))
        plt.close()


def plot_schedule_chunk_bars(df_omp_medians: pd.DataFrame, dataset: str, outdir: str):
    """Barra comparando schedules e chunks em um T fixo."""
    os.makedirs(outdir, exist_ok=True)

    Tfixed = pick_threads_fixed_available(df_omp_medians[df_omp_medians["dataset"] == dataset])
    if Tfixed is None:
        return

    d = df_omp_medians[
        (df_omp_medians["dataset"] == dataset) &
        (df_omp_medians["threads"] == Tfixed)
    ].copy()
    if d.empty:
        return

    d["label"] = d.apply(
        lambda r: f"{r['schedule']}" if r["chunk"] == "NA" else f"{r['schedule']},{r['chunk']}",
        axis=1
    )
    d = d.sort_values(["schedule", "chunk"])

    plt.figure(figsize=(10, 4))
    plt.bar(d["label"], d["median_ms"])
    plt.ylabel("Tempo (ms)")
    plt.title(f"OpenMP — Schedule/Chunk (T={Tfixed}) — dataset {dataset}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_omp_sched_chunk_T{Tfixed}.png"))
    plt.close()


def make_validation_report(df: pd.DataFrame, outpath: str):
    """Valida SSE:
       (1) estabilidade em repetições por configuração seq/omp
       (2) igualdade seq vs omp (medianas por dataset dentro da tolerância)
    """
    lines = []
    lines.append("VALIDACAO (SSE FINAL)\n")
    lines.append(f"Tolerancia usada: {TOL_SSE:e}\n")

    def keycols(mode):
        if mode == "seq":
            return ["dataset", "modo"]
        else:
            return ["dataset", "modo", "threads", "schedule", "chunk"]

    for mode in ["seq", "omp"]:
        cfg_cols = keycols(mode)
        df_reps = df[(df["modo"] == mode) & (df["rep"] > 0)].copy()
        if df_reps.empty:
            continue

        lines.append(f"\n(1) Estabilidade entre repeticoes — {mode}\n")
        for cfg, grp in df_reps.groupby(cfg_cols):
            sse_vals = grp["sse_final"].dropna().values
            if len(sse_vals) == 0:
                continue
            sse_min, sse_max = np.min(sse_vals), np.max(sse_vals)
            ok = (sse_max - sse_min) <= TOL_SSE
            lines.append(f"  {cfg}: min={sse_min:.6f} max={sse_max:.6f} | ok={ok}")

    lines.append("\n(2) Igualdade seq vs omp (por dataset, usando medianas de SSE final)\n")
    seq_med = df[(df["modo"] == "seq") & (df["rep"] > 0)].groupby("dataset")["sse_final"].median()
    omp_med = df[(df["modo"] == "omp") & (df["rep"] > 0)].groupby("dataset")["sse_final"].median()

    for ds in sorted(df["dataset"].dropna().unique()):
        s_seq = seq_med.get(ds, np.nan)
        s_omp = omp_med.get(ds, np.nan)
        if not (np.isnan(s_seq) or np.isnan(s_omp)):
            diff = abs(s_seq - s_omp)
            ok = diff <= TOL_SSE
            lines.append(
                f"  dataset={ds}: seq_med={s_seq:.6f} omp_med={s_omp:.6f} | |diff|={diff:.3e} | ok={ok}"
            )

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ===================== FUNÇÕES PARA CUDA =====================

def prepare_cuda_summary(df: pd.DataFrame, df_med: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """
    Monta tabela de medianas CUDA com:
      - block, grid
      - tempos: h2d_ms, kernel_ms, d2h_ms, total_ms
      - throughput_pts_s (pontos/s)
      - speedup_seq (vs sequencial)
    """
    df_cuda = df_med[df_med["modo"] == "cuda"].copy()
    if df_cuda.empty:
        return df_cuda

    df_cuda = df_cuda.merge(base, on="dataset", how="left")  # adiciona tempo_serial_ms
    df_cuda["N"] = df_cuda["dataset"].map(DATASET_N).astype(float)
    df_cuda["grid"] = np.ceil(df_cuda["N"] / df_cuda["block"])

    # Throughput = N / (total_ms/1000)
    df_cuda["throughput_pts_s"] = np.where(
        df_cuda["total_ms"] > 0,
        df_cuda["N"] * 1000.0 / df_cuda["total_ms"],
        np.nan,
    )

    df_cuda["speedup_seq"] = df_cuda["tempo_serial_ms"] / df_cuda["total_ms"]

    return df_cuda


def plot_cuda_times(df_cuda: pd.DataFrame, dataset: str, outdir: str):
    """Gráfico de tempos CUDA (H2D, Kernel, D2H, Total) vs block."""
    os.makedirs(outdir, exist_ok=True)
    d = df_cuda[df_cuda["dataset"] == dataset].copy()
    if d.empty:
        return

    d = d.sort_values("block")

    plt.figure()
    plt.plot(d["block"], d["h2d_ms"], marker="o", label="H2D (ms)")
    plt.plot(d["block"], d["kernel_ms"], marker="o", label="Kernel (ms)")
    plt.plot(d["block"], d["d2h_ms"], marker="o", label="D2H (ms)")
    plt.plot(d["block"], d["total_ms"], marker="o", linestyle="--", label="Total (ms)")
    plt.xlabel("Block size (threads por bloco)")
    plt.ylabel("Tempo (ms)")
    plt.title(f"CUDA — Tempos vs Block — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_cuda_tempos_vs_block.png"))
    plt.close()


def plot_cuda_throughput_speedup(df_cuda: pd.DataFrame, dataset: str, outdir: str):
    """Gráficos de throughput e speedup vs block (comparando com sequencial)."""
    os.makedirs(outdir, exist_ok=True)
    d = df_cuda[df_cuda["dataset"] == dataset].copy()
    if d.empty:
        return

    d = d.sort_values("block")

    # Throughput
    plt.figure()
    plt.plot(d["block"], d["throughput_pts_s"], marker="o")
    plt.xlabel("Block size (threads por bloco)")
    plt.ylabel("Throughput (pontos/s)")
    plt.title(f"CUDA — Throughput vs Block — dataset {dataset}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_cuda_throughput_vs_block.png"))
    plt.close()

    # Speedup vs seq
    plt.figure()
    plt.plot(d["block"], d["speedup_seq"], marker="o", label="Speedup vs seq")
    plt.xlabel("Block size (threads por bloco)")
    plt.ylabel("Speedup (vs sequencial)")
    plt.title(f"CUDA — Speedup vs Block — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_cuda_speedup_vs_block.png"))
    plt.close()


# ===================== COMPARAÇÃO GLOBAL (SEQ x OMP x CUDA) =====================

def make_global_comparison(base: pd.DataFrame,
                           df_omp_med: pd.DataFrame | None,
                           df_cuda_med: pd.DataFrame | None,
                           outdir_root: str) -> None:
    """
    Gera um CSV comparando:
      - sequencial
      - melhor configuração OpenMP por dataset
      - todas as configurações CUDA (por block)
    Com:
      - tempo (ms)
      - throughput (pontos/s)
      - speedup vs seq
      - speedup vs melhor OMP (para CUDA)
    """
    if df_omp_med is None or df_cuda_med is None:
        return
    if df_omp_med.empty or df_cuda_med.empty:
        return

    # Baseline seq: tempo e throughput
    base = base.copy()
    base["N"] = base["dataset"].map(DATASET_N).astype(float)
    base["throughput_pts_s"] = np.where(
        base["tempo_serial_ms"] > 0,
        base["N"] * 1000.0 / base["tempo_serial_ms"],
        np.nan
    )

    # Melhor OMP por dataset (menor median_ms)
    df_omp_med = df_omp_med.copy()
    best_omp = (
        df_omp_med
        .sort_values(["dataset", "median_ms"])
        .groupby("dataset", as_index=False)
        .first()
    )
    best_omp = best_omp.rename(columns={"median_ms": "tempo_omp_ms"})
    best_omp["N"] = best_omp["dataset"].map(DATASET_N).astype(float)
    best_omp["throughput_pts_s"] = np.where(
        best_omp["tempo_omp_ms"] > 0,
        best_omp["N"] * 1000.0 / best_omp["tempo_omp_ms"],
        np.nan
    )

    best_omp = best_omp.merge(
        df_omp_med[["dataset", "threads", "schedule", "chunk", "median_ms", "speedup"]],
        on=["dataset", "threads", "schedule", "chunk"],
        how="left",
        suffixes=("", "_chk")
    )
    best_omp["speedup_seq"] = best_omp["speedup"]

    df_cuda = df_cuda_med.copy()
    df_cuda = df_cuda.merge(
        best_omp[["dataset", "tempo_omp_ms"]],
        on="dataset", how="left"
    )
    df_cuda["speedup_omp"] = df_cuda["tempo_omp_ms"] / df_cuda["total_ms"]

    # Monta tabela final
    rows = []
    all_datasets = sorted(set(base["dataset"].dropna().unique())
                          | set(best_omp["dataset"].dropna().unique())
                          | set(df_cuda["dataset"].dropna().unique()))

    for ds in all_datasets:
        # Sequencial
        b = base[base["dataset"] == ds]
        if not b.empty:
            r = b.iloc[0]
            rows.append({
                "dataset": ds,
                "modo": "seq",
                "config": "seq",
                "threads": np.nan,
                "block": np.nan,
                "schedule": "NA",
                "chunk": "NA",
                "tempo_ms": r["tempo_serial_ms"],
                "throughput_pts_s": r["throughput_pts_s"],
                "speedup_seq": 1.0,
                "speedup_omp": np.nan,
            })

        # Melhor OMP
        o = best_omp[best_omp["dataset"] == ds]
        if not o.empty:
            r = o.iloc[0]
            cfg_label = f"omp_T{int(r['threads'])}_{r['schedule']}"
            if r["chunk"] != "NA":
                cfg_label += f",chunk={r['chunk']}"
            rows.append({
                "dataset": ds,
                "modo": "omp",
                "config": cfg_label,
                "threads": r["threads"],
                "block": np.nan,
                "schedule": r["schedule"],
                "chunk": r["chunk"],
                "tempo_ms": r["tempo_omp_ms"],
                "throughput_pts_s": r["throughput_pts_s"],
                "speedup_seq": r["speedup_seq"],
                "speedup_omp": np.nan,
            })

        # Todas as configs CUDA (por block)
        c = df_cuda[df_cuda["dataset"] == ds].sort_values("block")
        for _, r in c.iterrows():
            cfg_label = f"cuda_block{int(r['block'])}"
            rows.append({
                "dataset": ds,
                "modo": "cuda",
                "config": cfg_label,
                "threads": np.nan,
                "block": r["block"],
                "schedule": "NA",
                "chunk": "NA",
                "tempo_ms": r["total_ms"],
                "throughput_pts_s": r["throughput_pts_s"],
                "speedup_seq": r["speedup_seq"],
                "speedup_omp": r["speedup_omp"],
            })

    df_global = pd.DataFrame(rows)
    outdir = os.path.join(outdir_root, "global")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "comparacao_seq_omp_cuda.csv")
    df_global.to_csv(out_path, index=False)

    print("\n[GLOBAL] Comparação seq x melhor OMP x CUDA gerada.")
    print(f"CSV: {out_path}")
    print("Colunas: dataset, modo, config, tempo_ms, throughput_pts_s, speedup_seq, speedup_omp")


# ===================== MAIN =====================

def main():
    if len(sys.argv) < 3:
        print("Uso:")
        print("  python3 analisar_bench.py resultados_XXXX.csv openmp")
        print("  python3 analisar_bench.py resultados_XXXX.csv cuda")
        print("  python3 analisar_bench.py resultados_XXXX.csv all")
        sys.exit(1)

    csv_path = sys.argv[1]
    mode = sys.argv[2].lower()

    if mode not in ("openmp", "cuda", "all"):
        print("Modo inválido. Use 'openmp', 'cuda' ou 'all'.")
        sys.exit(1)

    df = load_csv(csv_path)

    # Linhas-resumo (rep = 0) para medianas
    df_med = df[df["rep"] == 0].copy()

    # Baselines sequenciais
    base = summarize_baselines(df)

    # Diretórios de saída
    root_outdir = "figs_bench"
    os.makedirs(root_outdir, exist_ok=True)

    df_omp_med = None
    df_cuda_med = None

    if mode in ("openmp", "all"):
        outdir_omp = os.path.join(root_outdir, "openmp")
        os.makedirs(outdir_omp, exist_ok=True)

        # ====== ANALISE OPENMP ======
        df_omp_med = df_med[df_med["modo"] == "omp"].copy()
        if df_omp_med.empty:
            print("Não há linhas com modo='omp' (OpenMP) no CSV.")
        else:
            df_omp_med = compute_speedup_openmp(df_omp_med, base)

            # Gráficos por dataset
            for ds in sorted(df_omp_med["dataset"].dropna().unique()):
                plot_time_speedup_throughput_openmp(df_omp_med, ds, outdir_omp)
                plot_schedule_chunk_bars(df_omp_med, ds, outdir_omp)

            # Relatório de validação de SSE (usa reps individuais)
            make_validation_report(df, os.path.join(outdir_omp, "validacao_sse.txt"))

            # Tabelas-síntese
            curves = (
                df_omp_med
                .loc[:, ["dataset", "threads", "schedule", "chunk", "median_ms", "speedup"]]
                .sort_values(["dataset", "schedule", "chunk", "threads"])
            )
            curves.to_csv(os.path.join(outdir_omp, "openmp_curvas_tempo_speedup.csv"), index=False)

            best_per_T = (
                df_omp_med
                .sort_values(["dataset", "threads", "median_ms"])
                .groupby(["dataset", "threads"], as_index=False)
                .first()
                .loc[:, ["dataset", "threads", "schedule", "chunk", "median_ms", "speedup"]]
            )
            best_per_T.to_csv(os.path.join(outdir_omp, "openmp_melhor_config_por_threads.csv"), index=False)

            print("\nConcluído (modo OPENMP).")
            print(f"- Figuras em: {outdir_omp}/*.png")
            print(f"- Validação SSE: {outdir_omp}/validacao_sse.txt")
            print(f"- Tabelas: {outdir_omp}/openmp_curvas_tempo_speedup.csv e openmp_melhor_config_por_threads.csv")

    if mode in ("cuda", "all"):
        outdir_cuda = os.path.join(root_outdir, "cuda")
        os.makedirs(outdir_cuda, exist_ok=True)

        # ====== ANALISE CUDA ======
        df_cuda_med = prepare_cuda_summary(df, df_med, base)
        if df_cuda_med.empty:
            print("Não há linhas com modo='cuda' no CSV (rep=0).")
        else:
            # Tabela consolidada
            cols_resumo = [
                "dataset", "block", "grid",
                "h2d_ms", "kernel_ms", "d2h_ms", "total_ms",
                "throughput_pts_s", "speedup_seq"
            ]
            df_cuda_med[cols_resumo].sort_values(["dataset", "block"]).to_csv(
                os.path.join(outdir_cuda, "cuda_resumo.csv"), index=False
            )

            # Gráficos por dataset
            for ds in sorted(df_cuda_med["dataset"].dropna().unique()):
                plot_cuda_times(df_cuda_med, ds, outdir_cuda)
                plot_cuda_throughput_speedup(df_cuda_med, ds, outdir_cuda)

            print("\nConcluído (modo CUDA).")
            print(f"- Figuras em: {outdir_cuda}/*cuda_*.png")
            print(f"- Tabela consolidada: {outdir_cuda}/cuda_resumo.csv")
            print("  (contém block, grid, tempos H2D/Kernel/D2H/Total, throughput e speedup vs seq)")

    # Comparação global (somente se 'all')
    if mode == "all":
        make_global_comparison(base, df_omp_med, df_cuda_med, root_outdir)


if __name__ == "__main__":
    main()