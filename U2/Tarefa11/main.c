#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stddef.h>

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
    FILE *arquivo = fopen("resultados.csv", "w");

    if (arquivo == NULL) {
        return EXIT_FAILURE;
    }

    fprintf(arquivo, "tamanho_vetor,executado_por,tempo\n");

    for (size_t i = 0; i < NUM_TAMANHOS; i++) {
        size_t tamanho = tamanhos[i];

        Vetores *v = criarVetor(tamanho);

        double tempoCPU = calcularCPU(v, tamanho);

        if (!verificarResultado(zv, tamanho)) {
            liberarVetores(v);
            fclose(arquivo);
            return EXIT_FAILURE;
        }

        fprintf(arquivo, "%zu,cpu,%.10f\n", tamanho, tempoCPU);

        double tempoGPU = calcularGPU(v, tamanho);

        if (!verificarResultado(v, tamanho)) {
            liberarVetores(v);
            fclose(arquivo);
            return EXIT_FAILURE;
        }

        fprintf(arquivo, "%zu,gpu,%.10f\n", tamanho, tempoGPU);

        printf("%zu: CPU = %.10f s | GPU = %.10f s\n",
               tamanho, tempoCPU, tempoGPU);

        liberarVetores(v);
    }

    fclose(arquivo);

    return EXIT_SUCCESS;
}
