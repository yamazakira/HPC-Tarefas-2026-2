#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define REPETICOES 10
#define ARQUIVO_JSON "resultados_cuda.json"

#define CUDA_CHECK(chamada)                                              \
    do {                                                                 \
        cudaError_t erro_ = (chamada);                                   \
        if (erro_ != cudaSuccess) {                                      \
            fprintf(stderr, "Erro CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(erro_));                          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

static const size_t tamanhos[] = {
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    250000000,
};
static const size_t quantidade_tamanhos =
    sizeof(tamanhos) / sizeof(tamanhos[0]);

static const int blocos_testados[] = {32, 64, 128, 256, 512, 1024};
static const size_t quantidade_blocos =
    sizeof(blocos_testados) / sizeof(blocos_testados[0]);

static const char *classe_tamanho(size_t n)
{
    if (n <= 10000) return "pequeno";
    if (n <= 1000000) return "medio";
    return "grande";
}

__global__ void soma_divergente(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t passo = (size_t)gridDim.x * blockDim.x;

    for (; i < n; i += passo) {
        if (i % 2 == 0) {
            C[i] = 2.0f * A[i] + B[i];
        } else {
            C[i] = A[i] + 3.0f * B[i];
        }
    }
}

__global__ void soma_sem_desvio(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t passo = (size_t)gridDim.x * blockDim.x;

    for (; i < n; i += passo) {
        float impar = (float)(i & 1);
        C[i] = (2.0f - impar) * A[i] + (1.0f + 2.0f * impar) * B[i];
    }
}

static void soma_cpu(const float *A, const float *B, float *C, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (i % 2 == 0) {
            C[i] = 2.0f * A[i] + B[i];
        } else {
            C[i] = A[i] + 3.0f * B[i];
        }
    }
}

static double tempo_atual(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static size_t verifica(const float *ref, const float *gpu, size_t n)
{
    size_t erros = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(ref[i] - gpu[i]);
        if (d > 1e-4f * fmaxf(1.0f, fabsf(ref[i]))) erros++;
    }
    return erros;
}

typedef void (*KernelSoma)(const float *, const float *, float *, size_t);

static float mede_kernel(KernelSoma kernel, int grid, int bloco,
                         const float *dA, const float *dB, float *dC,
                         size_t n, cudaEvent_t ini, cudaEvent_t fim)
{

    kernel<<<grid, bloco>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float soma = 0.0f;
    for (int r = 0; r < REPETICOES; r++) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ini));
        kernel<<<grid, bloco>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaEventRecord(fim));
        CUDA_CHECK(cudaEventSynchronize(fim));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ini, fim));
        soma += ms;
    }
    return soma / REPETICOES;
}

static double ocupacao_teorica(KernelSoma kernel, int bloco,
                               const cudaDeviceProp *prop, int *blocos_sm)
{
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocos_sm, kernel, bloco, 0));
    return 100.0 * (*blocos_sm * bloco) / prop->maxThreadsPerMultiProcessor;
}

static int primeiro_registro = 1;

static void escreve_registro(FILE *json, size_t n, const char *config,
                             const char *kernel, int bloco, int grid,
                             double ocupacao, float tempo_ms, size_t erros)
{
    double bytes = 3.0 * n * sizeof(float);
    double banda = bytes / (tempo_ms * 1e-3) / 1e9;

    fprintf(json,
            "%s    {\"n\": %zu, \"classe\": \"%s\", \"config\": \"%s\", "
            "\"kernel\": \"%s\", \"bloco\": %d, \"grid\": %d, "
            "\"ocupacao_teorica_pct\": %.2f, \"tempo_kernel_ms\": %f, "
            "\"banda_gbs\": %f, \"erros\": %zu}",
            primeiro_registro ? "" : ",\n", n, classe_tamanho(n), config,
            kernel, bloco, grid, ocupacao, tempo_ms, banda, erros);
    primeiro_registro = 0;

    printf("  %-22s %-12s bloco %4d grid %9d | ocup %6.2f%% | "
           "%10.4f ms | %8.2f GB/s | erros %zu\n",
           config, kernel, bloco, grid, ocupacao, tempo_ms, banda, erros);
}

int main(void)
{

    int dispositivo = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dispositivo));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dispositivo));

    printf("GPU: %s | CC %d.%d | SMs: %d | max threads/SM: %d | "
           "max threads/bloco: %d | warp: %d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock,
           prop.warpSize);

    int grid_min = 0, bloco_ocupacao = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &grid_min, &bloco_ocupacao, soma_sem_desvio, 0, 0));

    int blocos_sm_ocupacao = 0;
    double ocup_api = ocupacao_teorica(soma_sem_desvio, bloco_ocupacao,
                                       &prop, &blocos_sm_ocupacao);

    int grid_dispositivo = prop.multiProcessorCount * blocos_sm_ocupacao;

    printf("API de ocupacao: bloco = %d, blocos ativos/SM = %d "
           "(ocupacao %.1f%%), grid persistente = %d blocos (%d SMs x %d)\n\n",
           bloco_ocupacao, blocos_sm_ocupacao, ocup_api, grid_dispositivo,
           prop.multiProcessorCount, blocos_sm_ocupacao);

    size_t n_max = 0;
    for (size_t t = 0; t < quantidade_tamanhos; t++)
        if (tamanhos[t] > n_max) n_max = tamanhos[t];

    size_t bytes_max = n_max * sizeof(float);

    float *hA, *hB, *hC, *hRef;
    CUDA_CHECK(cudaMallocHost(&hA, bytes_max));
    CUDA_CHECK(cudaMallocHost(&hB, bytes_max));
    CUDA_CHECK(cudaMallocHost(&hC, bytes_max));
    hRef = (float *)malloc(bytes_max);
    if (hRef == NULL) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        return 1;
    }

    for (size_t i = 0; i < n_max; i++) {
        hA[i] = (float)(i % 1000) * 0.5f;
        hB[i] = (float)(i % 777) * 0.25f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes_max));
    CUDA_CHECK(cudaMalloc(&dB, bytes_max));
    CUDA_CHECK(cudaMalloc(&dC, bytes_max));

    cudaEvent_t ini, fim;
    CUDA_CHECK(cudaEventCreate(&ini));
    CUDA_CHECK(cudaEventCreate(&fim));

    FILE *json = fopen(ARQUIVO_JSON, "w");
    if (json == NULL) {
        fprintf(stderr, "Erro ao abrir %s\n", ARQUIVO_JSON);
        return 1;
    }

    fprintf(json, "{\n");
    fprintf(json, "  \"gpu\": \"%s\",\n", prop.name);
    fprintf(json, "  \"compute_capability\": \"%d.%d\",\n", prop.major,
            prop.minor);
    fprintf(json, "  \"sms\": %d,\n", prop.multiProcessorCount);
    fprintf(json, "  \"max_threads_por_sm\": %d,\n",
            prop.maxThreadsPerMultiProcessor);
    fprintf(json, "  \"max_threads_por_bloco\": %d,\n",
            prop.maxThreadsPerBlock);
    fprintf(json, "  \"warp\": %d,\n", prop.warpSize);
    fprintf(json, "  \"bloco_api_ocupacao\": %d,\n", bloco_ocupacao);
    fprintf(json, "  \"grid_persistente\": %d,\n", grid_dispositivo);
    fprintf(json, "  \"ocupacao_api_pct\": %.2f,\n", ocup_api);
    fprintf(json, "  \"repeticoes\": %d,\n", REPETICOES);

    fprintf(json, "  \"base\": [\n");

    for (size_t t = 0; t < quantidade_tamanhos; t++) {
        size_t n = tamanhos[t];
        size_t bytes = n * sizeof(float);

        double t0 = tempo_atual();
        soma_cpu(hA, hB, hRef, n);
        double tempo_cpu_ms = (tempo_atual() - t0) * 1e3;

        float h2d = 0.0f, d2h = 0.0f;
        CUDA_CHECK(cudaEventRecord(ini));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(fim));
        CUDA_CHECK(cudaEventSynchronize(fim));
        CUDA_CHECK(cudaEventElapsedTime(&h2d, ini, fim));

        CUDA_CHECK(cudaEventRecord(ini));
        CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(fim));
        CUDA_CHECK(cudaEventSynchronize(fim));
        CUDA_CHECK(cudaEventElapsedTime(&d2h, ini, fim));

        fprintf(json,
                "    {\"n\": %zu, \"classe\": \"%s\", \"tempo_cpu_ms\": %f, "
                "\"tempo_h2d_ms\": %f, \"tempo_d2h_ms\": %f}%s\n",
                n, classe_tamanho(n), tempo_cpu_ms, h2d, d2h,
                t + 1 < quantidade_tamanhos ? "," : "");
    }
    fprintf(json, "  ],\n");

    fprintf(json, "  \"resultados\": [\n");

    for (size_t t = 0; t < quantidade_tamanhos; t++) {
        size_t n = tamanhos[t];
        size_t bytes = n * sizeof(float);

        printf("====== N = %zu (%s) ======\n", n, classe_tamanho(n));

        soma_cpu(hA, hB, hRef, n);
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

        for (size_t b = 0; b < quantidade_blocos; b++) {
            int bloco = blocos_testados[b];
            int grid = (int)((n + bloco - 1) / bloco);
            int blocos_sm;

            for (int k = 0; k < 2; k++) {
                KernelSoma kernel = k == 0 ? soma_sem_desvio
                                           : soma_divergente;
                double ocup = ocupacao_teorica(kernel, bloco, &prop,
                                               &blocos_sm);
                CUDA_CHECK(cudaMemset(dC, 0, bytes));
                float ms = mede_kernel(kernel, grid, bloco, dA, dB, dC, n,
                                       ini, fim);
                CUDA_CHECK(cudaMemcpy(hC, dC, bytes,
                                      cudaMemcpyDeviceToHost));
                escreve_registro(json, n, "uma_thread_por_elemento",
                                 k == 0 ? "sem_desvio" : "divergente",
                                 bloco, grid, ocup, ms,
                                 verifica(hRef, hC, n));
            }
        }

        {
            int bloco = bloco_ocupacao;
            int grid_necessario = (int)((n + bloco - 1) / bloco);
            int grid = grid_dispositivo < grid_necessario ? grid_dispositivo
                                                          : grid_necessario;
            for (int k = 0; k < 2; k++) {
                KernelSoma kernel = k == 0 ? soma_sem_desvio
                                           : soma_divergente;
                int blocos_sm;
                double ocup = ocupacao_teorica(kernel, bloco, &prop,
                                               &blocos_sm);
                CUDA_CHECK(cudaMemset(dC, 0, bytes));
                float ms = mede_kernel(kernel, grid, bloco, dA, dB, dC, n,
                                       ini, fim);
                CUDA_CHECK(cudaMemcpy(hC, dC, bytes,
                                      cudaMemcpyDeviceToHost));
                escreve_registro(json, n, "dispositivo_grid_stride",
                                 k == 0 ? "sem_desvio" : "divergente",
                                 bloco, grid, ocup, ms,
                                 verifica(hRef, hC, n));
            }
        }

        {
            int bloco = 256;
            int grid = prop.multiProcessorCount;
            double ocup = 100.0 * bloco / prop.maxThreadsPerMultiProcessor;
            CUDA_CHECK(cudaMemset(dC, 0, bytes));
            float ms = mede_kernel(soma_sem_desvio, grid, bloco, dA, dB, dC,
                                   n, ini, fim);
            CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
            escreve_registro(json, n, "um_bloco_por_sm", "sem_desvio",
                             bloco, grid, ocup, ms, verifica(hRef, hC, n));
        }
    }

    fprintf(json, "\n  ]\n}\n");
    fclose(json);

    printf("\nResultados salvos em %s\n", ARQUIVO_JSON);

    CUDA_CHECK(cudaEventDestroy(ini));
    CUDA_CHECK(cudaEventDestroy(fim));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    free(hRef);

    return 0;
}
