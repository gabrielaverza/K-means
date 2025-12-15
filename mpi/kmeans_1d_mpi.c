/* kmeans_1d_mpi.c
   Etapa 3 — MPI
   - distribui X entre P processos; centróides C são globais a cada iteração
   - itera: broadcast inicial de C -> assignment local -> reduções globais
   - mede tempo total e tempo de comunicação (Allreduce)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- decomposição 1D: distribui N pontos entre P processos ---------- */
static void decompose_1d(int N, int P, int *counts, int *displs){
    int base  = N / P;
    int resto = N % P;
    int offset = 0;
    for(int p=0; p<P; p++){
        counts[p] = base + (p < resto ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }
}

/* ---------- k-means 1D ---------- */
/* assignment: pontos dividios entre os processos, assigment somente nos dados locais e calcula sse parcial */
static double assignment_step_1d_mpi(const double *X_local, int local_N, const double *C, int K, int *assign_local, double *sum_local, int *cnt_local){
    /* zera acumuladores locais */
    for(int c=0; c<K; c++){
        sum_local[c] = 0.0;
        cnt_local[c] = 0;
    }
    double sse_local = 0.0;
    for(int i=0; i<local_N; i++){
        double xi = X_local[i];
        int best = -1;
        double bestd = 1.0e300;
        for(int c=0;c<K;c++){
            double diff = xi - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign_local[i] = best;
        sse_local += bestd;
        sum_local[best] += xi;
        cnt_local[best] += 1;
    }
    return sse_local;
}

/* reducao e update: usa MPI_Reduce para o sse e MPI_Allreduce para soma e contagem por cluster */
static void update_step_1d_mpi(double *C, int K, double X0, double sse_local, double *sse_global, double *sum_local, int *cnt_local, int rank, MPI_Comm comm, double *comm_time_accum){
    MPI_Reduce(&sse_local, sse_global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    /* aloca vetores para somas e contagens globais */
    double *sum_global = (double*)calloc((size_t)K, sizeof(double));
    int *cnt_global = (int*)calloc((size_t)K, sizeof(int));
    if(!sum_global || !cnt_global){ fprintf(stderr,"Erro: sem memoria em update_step_1d_mpi\n"); MPI_Abort(comm, 1); }

    /* mede tempo de comunicação (Allreduce) */
    double tcomm0 = MPI_Wtime();

    MPI_Allreduce(sum_local, sum_global, K, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(cnt_local, cnt_global, K, MPI_INT, MPI_SUM, comm);

    double tcomm1 = MPI_Wtime();
    *comm_time_accum += (tcomm1 - tcomm0);

    for(int c=0; c<K; c++){
        if(cnt_global[c] > 0){
            C[c] = sum_global[c] / (double)cnt_global[c];
        }else{
            /* cluster vazio: copia X0 */
            C[c] = X0;
        }
    }

    free(sum_global);
    free(cnt_global);
}

static void kmeans_1d_mpi(double *X_global, double *C, int *assign_global, int N, int K, int max_iter, double eps, int rank, int nprocs, int *iters_out, double *sse_out, double *comm_time_out){
    MPI_Comm comm = MPI_COMM_WORLD;

    /* decomposição do domínio */
    int *counts = (int*)malloc((size_t)nprocs * sizeof(int));
    int *displs = (int*)malloc((size_t)nprocs * sizeof(int));
    if(!counts || !displs){
        fprintf(stderr,"Rank %d: sem memoria para counts/displs\n", rank);
        MPI_Abort(comm, 1);
    }
    decompose_1d(N, nprocs, counts, displs);
    int local_N = counts[rank];

    /* buffers locais para dados e assign */
    double *X_local = (double*)malloc((size_t)local_N * sizeof(double));
    int    *assign_local = (int*)malloc((size_t)local_N * sizeof(int));
    if(!X_local || !assign_local){
        fprintf(stderr,"Rank %d: sem memoria em X_local/assign_local\n", rank);
        MPI_Abort(comm, 1);
    }

    /* scatter: distribui blocos de X_global para X_local de cada processo */
    MPI_Scatterv(X_global, counts, displs, MPI_DOUBLE, X_local, local_N, MPI_DOUBLE, 0, comm);

    /* X0 global para tratar clusters vazios */
    double X0 = 0.0;
    if(rank == 0){
        X0 = X_global[0];
    }
    MPI_Bcast(&X0, 1, MPI_DOUBLE, 0, comm);

    /* broadcast inicial de C */
    MPI_Bcast(C, K, MPI_DOUBLE, 0, comm);

    double prev_sse = 1e300;
    double sse_global = 0.0;
    int it;
    int stop = 0;
    double comm_time = 0.0; /* acumulador de tempo dos Allreduce */

    for(it = 0; it < max_iter; it++){
        /* aloca acumuladores locais de cada iteração */
        double *sum_local = (double*)malloc((size_t)K * sizeof(double));
        int *cnt_local = (int*)malloc((size_t)K * sizeof(int));
        if(!sum_local || !cnt_local){ fprintf(stderr,"Rank %d: sem memoria para sum_local/cnt_local\n", rank); MPI_Abort(comm, 1); }

        double sse_local = assignment_step_1d_mpi(X_local, local_N, C, K, assign_local, sum_local, cnt_local);
        update_step_1d_mpi(C, K, X0, sse_local, &sse_global, sum_local, cnt_local, rank, comm, &comm_time);

        free(sum_local);
        free(cnt_local);

        /* critério de parada: rank 0 decide, depois broadcast de 'stop' */
        if(rank == 0){
            double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
            if(rel < eps){
                stop = 1;
            }else{
                stop = 0;
                prev_sse = sse_global;
            }
        }

        MPI_Bcast(&stop, 1, MPI_INT, 0, comm);

        if(stop){
            it++; /* conta a iteração de convergência */
            break;
        }
    }

    /* coleta final de assigns no rank 0 */
    MPI_Gatherv(assign_local, local_N, MPI_INT, assign_global, counts, displs, MPI_INT, 0, comm);

    free(X_local);
    free(assign_local);
    free(counts);
    free(displs);

    if(rank == 0){
        *iters_out      = it;
        *sse_out        = sse_global;
        *comm_time_out  = comm_time;
    }
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc < 3){
        if(rank == 0){
            printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3) ? atoi(argv[3]) : 50;
    double eps   = (argc>4) ? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5) ? argv[5] : NULL;
    const char *outCentroid = (argc>6) ? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        if(rank == 0)
            fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        MPI_Finalize();
        return 1;
    }

    int N=0, K=0;
    double *X_global = NULL;
    double *C = NULL;
    int *assign_global = NULL;

    if(rank == 0){
        X_global = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign_global = (int*)malloc((size_t)N * sizeof(int));
        if(!assign_global){
            fprintf(stderr,"Rank 0: sem memoria para assign_global\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* broadcast dos parâmetros globais */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* todos os processos precisam ter espaço para C */
    if(rank != 0){
        C = (double*)malloc((size_t)K * sizeof(double));
        if(!C){
            fprintf(stderr,"Rank %d: sem memoria para C\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* medição: tempo de parede */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int iters = 0;
    double sse_final = 0.0;
    double comm_time = 0.0;

    kmeans_1d_mpi(X_global, C, assign_global, N, K, max_iter, eps, rank, nprocs, &iters, &sse_final, &comm_time);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double total_ms = (t1 - t0) * 1000.0;
    double comm_ms  = comm_time * 1000.0;

    if(rank == 0){
        printf("K-means 1D (MPI)\n");
        printf("N=%d  K=%d  max_iter=%d  eps=%g  P=%d\n",
               N, K, max_iter, eps, nprocs);
        printf("Iterações: %d\n", iters);
        printf("SSE final: %.6f\n", sse_final);
        printf("Tempo total (apenas K-means): %.3f ms\n", total_ms);
        printf("Tempo de comunicação (Allreduce): %.3f ms\n", comm_ms);
        printf("Tempo de computação aproximado: %.3f ms\n", total_ms - comm_ms);
        printf("Para Strong Scaling, execute com P = 1, 2, 4, 8, ... e compare o tempo total.\n");
        printf("Para Speedup, use: Speedup(P) = Tempo_serial_naive / Tempo_MPI(P).\n");

        write_assign_csv(outAssign,   assign_global, N);
        write_centroids_csv(outCentroid, C, K);

        free(X_global);
        free(C);
        free(assign_global);
    }else{
        free(C);
    }

    MPI_Finalize();
    return 0;
}