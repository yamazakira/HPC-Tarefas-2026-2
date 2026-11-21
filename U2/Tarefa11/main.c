#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stddef.h>
#include <string.h>

static const size_t tamanhos[] = {
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000
};

#define NUM_TAMANHOS (sizeof(tamanhos) / sizeof(tamanhos[0]))

typedef struct {
    float *A;
    float *B;
    float *C;
} Vetores;

Vetores* criarVetor(size_t tamanho)
{
    Vetores *v = malloc(sizeof(Vetores));

    if (v == NULL) {
        exit(EXIT_FAILURE);
    }

    v->A = malloc(tamanho * sizeof(float));
    v->B = malloc(tamanho * sizeof(float));
    v->C = malloc(tamanho * sizeof(float));

    if (v->A == NULL || v->B == NULL || v->C == NULL) {
        free(v->A);
        free(v->B);
        free(v->C);
        free(v);
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < tamanho; i++) {
        v->B[i] = (float)i;
        v->C[i] = (float)(i * 2);
        v->A[i] = 0.0f;
    }

    return v;
}

void liberarVetores(Vetores *v)
{
    free(v->A);
    free(v->B);
    free(v->C);
    free(v);
}

double calcularCPU(Vetores *v, size_t tamanho)
{
    double inicio = omp_get_wtime();

    #pragma omp parallel for
    for (size_t i = 0; i < tamanho; i++) {
        v->A[i] = v->B[i] + v->C[i];
    }

    return omp_get_wtime() - inicio;
}

double calcularGPU(Vetores *v, size_t tamanho)
{
    double inicio = omp_get_wtime();

    #pragma omp target teams distribute parallel for \
        map(to: v->B[0:tamanho], v->C[0:tamanho]) \
        map(from: v->A[0:tamanho])
    for (size_t i = 0; i < tamanho; i++) {
        v->A[i] = v->B[i] + v->C[i];
    }

    return omp_get_wtime() - inicio;
}

double calcularGPUSeparado(Vetores *v, size_t tamanho, double *transferencia)
{
    float *A = v->A;
    float *B = v->B;
    float *C = v->C;

    double inicio = omp_get_wtime();

    #pragma omp target enter data map(to: B[0:tamanho], C[0:tamanho]) \
        map(alloc: A[0:tamanho])

    double inicioKernel = omp_get_wtime();

    #pragma omp target teams distribute parallel for
    for (size_t i = 0; i < tamanho; i++) {
        A[i] = B[i] + C[i];
    }

    double tempoKernel = omp_get_wtime() - inicioKernel;

    #pragma omp target exit data map(from: A[0:tamanho]) \
        map(release: B[0:tamanho], C[0:tamanho])

    *transferencia = (omp_get_wtime() - inicio) - tempoKernel;

    return tempoKernel;
}

int verificarResultado(Vetores *v, size_t tamanho)
{
    for (size_t i = 0; i < tamanho; i++) {
        if (v->A[i] != v->B[i] + v->C[i]) {
            return 0;
        }
    }

    return 1;
}

int main(void)
{
    FILE *arquivo = fopen("resultados.json", "w");

    if (arquivo == NULL) {
        return EXIT_FAILURE;
    }

    #pragma omp target
    {
    }

    fprintf(arquivo, "{\n  \"dispositivos\": %d,\n  \"threads\": %d,\n  \"resultados\": [\n",
            omp_get_num_devices(), omp_get_max_threads());

    for (size_t i = 0; i < NUM_TAMANHOS; i++) {
        size_t tamanho = tamanhos[i];

        Vetores *v = criarVetor(tamanho);

        double tempoCPU = calcularCPU(v, tamanho);

        if (!verificarResultado(v, tamanho)) {
            liberarVetores(v);
            fclose(arquivo);
            return EXIT_FAILURE;
        }

        memset(v->A, 0, tamanho * sizeof(float));

        double tempoGPU = calcularGPU(v, tamanho);

        if (!verificarResultado(v, tamanho)) {
            liberarVetores(v);
            fclose(arquivo);
            return EXIT_FAILURE;
        }

        memset(v->A, 0, tamanho * sizeof(float));

        double tempoTransferencia = 0.0;
        double tempoKernel = calcularGPUSeparado(v, tamanho, &tempoTransferencia);

        if (!verificarResultado(v, tamanho)) {
            liberarVetores(v);
            fclose(arquivo);
            return EXIT_FAILURE;
        }

        fprintf(arquivo,
                "    {\"tamanho\": %zu, \"cpu_s\": %.10f, \"gpu_total_s\": %.10f, "
                "\"gpu_kernel_s\": %.10f, \"gpu_transferencia_s\": %.10f}%s\n",
                tamanho, tempoCPU, tempoGPU, tempoKernel, tempoTransferencia,
                i + 1 < NUM_TAMANHOS ? "," : "");

        printf("%zu: CPU = %.10f s | GPU = %.10f s (kernel %.10f s + transferencia %.10f s)\n",
               tamanho, tempoCPU, tempoGPU, tempoKernel, tempoTransferencia);

        liberarVetores(v);
    }

    fprintf(arquivo, "  ]\n}\n");
    fclose(arquivo);

    return EXIT_SUCCESS;
}
