/* kmeans_1d_cuda.cu
   Etapa 1 — CUDA
   - mover o assignment para a GPU; update feito no host (CPU)
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        bool only_ws=true;
        for(char *p=line; *p; ++p){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=false; break; }
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
        bool only_ws=true;
        for(char *p=line; *p; ++p){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=false; break; }
        }
        if(only_ws) continue;

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); exit(1); }
        A[r++] = atof(tok);
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

/* ---------- k-means 1D ---------- */
/* assignment: cada thread i varre todos os K centróides, escolhe o melhor, grava assign[i] e sse_per_point[i] */
__global__ void assignment_step_1d_kernel(const double* __restrict__ X,
                                  const double* __restrict__ C,
                                  int* __restrict__ assign,
                                  double* __restrict__ sse_per_point,
                                  int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    double xi = X[i];
    int best = -1;
    double bestd = 1.0e300;

    /* varre K centróides */
    for(int c=0;c<K;c++){
        double diff = xi - C[c];
        double d = diff * diff;
        if(d < bestd){ bestd = d; best = c; }
    }
    assign[i] = best;
    sse_per_point[i] = bestd;
}

/* host: update e outer loop */
static void update_step_1d_host(const double *X, double *C, const int *assign, int N, int K){
    std::vector<double> sum(K, 0.0);
    std::vector<int> cnt(K, 0);

    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; /* fallback simples para cluster vazio */
    }
}

static void kmeans_1d_cuda(const double *X_h, double *C_h, int *assign_h,
                           int N, int K, int max_iter, double eps,
                           int block_size,
                           int *iters_out, double *sse_out,
                           float *ms_h2d, float *ms_kernel, float *ms_d2h, double *ms_total)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    /* alocações no device */
    double *X_d=nullptr, *C_d=nullptr, *sse_d=nullptr;
    int *assign_d=nullptr;

    cudaEvent_t eH2D_start, eH2D_stop, eK_start, eK_stop, eD2H_start, eD2H_stop;
    cudaEventCreate(&eH2D_start); cudaEventCreate(&eH2D_stop);
    cudaEventCreate(&eK_start);   cudaEventCreate(&eK_stop);
    cudaEventCreate(&eD2H_start); cudaEventCreate(&eD2H_stop);

    cudaMalloc(&X_d,    (size_t)N * sizeof(double));
    cudaMalloc(&C_d,    (size_t)K * sizeof(double));
    cudaMalloc(&assign_d,(size_t)N * sizeof(int));
    cudaMalloc(&sse_d,  (size_t)N * sizeof(double));

    /* copia X e C pra GPU */
    cudaEventRecord(eH2D_start);
    cudaMemcpy(X_d, X_h, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C_h, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(eH2D_stop);
    cudaEventSynchronize(eH2D_stop);
    cudaEventElapsedTime(ms_h2d, eH2D_start, eH2D_stop);

    int grid = (N + block_size - 1) / block_size;

    double prev_sse = 1.0e300;
    double sse = 0.0;
    int it = 0;

    std::vector<double> sse_host(N);

    for(it=0; it<max_iter; ++it){
        /* assignment no kernel */
        cudaEventRecord(eK_start);
        assignment_step_1d_kernel<<<grid, block_size>>>(X_d, C_d, assign_d, sse_d, N, K);
        cudaEventRecord(eK_stop);
        cudaEventSynchronize(eK_stop);
        float this_kernel_ms = 0.0f;
        cudaEventElapsedTime(&this_kernel_ms, eK_start, eK_stop);
        *ms_kernel += this_kernel_ms;

        /* copia os resultados do assignment pro host (assign + sse por ponto) */
        cudaEventRecord(eD2H_start);
        cudaMemcpy(assign_h, assign_d, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(sse_host.data(), sse_d, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaEventRecord(eD2H_stop);
        cudaEventSynchronize(eD2H_stop);
        float this_d2h_ms = 0.0f;
        cudaEventElapsedTime(&this_d2h_ms, eD2H_start, eD2H_stop);
        *ms_d2h += this_d2h_ms;

        /* soma SSE no host */
        sse = 0.0;
        for(int i=0;i<N;i++) sse += sse_host[i];

        /* usa variação relativa da SSE no critério de parada */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        /*  ppdate no host */
        update_step_1d_host(X_h, C_h, assign_h, N, K);

        /* copia C atualizado pra GPU, pra próxima iteração */
        cudaMemcpy(C_d, C_h, (size_t)K * sizeof(double), cudaMemcpyHostToDevice);

        prev_sse = sse;
    }

    cudaFree(X_d); cudaFree(C_d); cudaFree(assign_d); cudaFree(sse_d);

    auto t1 = clock::now();
    *ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    *iters_out = it;
    *sse_out = sse;

    cudaEventDestroy(eH2D_start); cudaEventDestroy(eH2D_stop);
    cudaEventDestroy(eK_start);   cudaEventDestroy(eK_stop);
    cudaEventDestroy(eD2H_start); cudaEventDestroy(eD2H_stop);
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [block_size=256] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: CSVs 1 coluna (1 valor por linha), sem cabecalho.\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    int block_sz = (argc>5)? atoi(argv[5]) : 256;
    const char *outAssign   = (argc>6)? argv[6] : nullptr;
    const char *outCentroid = (argc>7)? argv[7] : nullptr;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parametros invalidos: max_iter>0 e eps>0\n");
        return 1;
    }
    if(block_sz <= 0) block_sz = 256;

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    int iters=0; double sse=0.0;
    float ms_h2d=0.0f, ms_kernel=0.0f, ms_d2h=0.0f;
    double ms_total=0.0;

    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, block_sz,
                   &iters, &sse, &ms_h2d, &ms_kernel, &ms_d2h, &ms_total);

    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g block=%d\n", N, K, max_iter, eps, block_sz);
    printf("Iteracoes: %d | SSE final: %.6f | Tempo total: %.1f ms\n", iters, sse, ms_total);
    printf("Tempos: H2D=%.3f ms | Kernel=%.3f ms | D2H=%.3f ms\n", ms_h2d, ms_kernel, ms_d2h);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}