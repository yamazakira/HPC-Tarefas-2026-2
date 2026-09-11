#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI_REF 3.14159265358979323846

typedef double (*func_t)(double);

static double f(double x) {
    return 4.0 / (1.0 + x * x);
}

double trapezio_sequencial(func_t fn, double a, double b, long n) {
    double h = (b - a) / (double) n;
    double soma = (fn(a) + fn(b)) / 2.0;

    for (long i = 1; i < n; i++) {
        soma += fn(a + i * h);
    }

    return soma * h;
}

double trapezio_paralelo_sem_protecao(func_t fn, double a, double b, long n) {
    double h = (b - a) / (double) n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++) {
        soma += fn(a + i * h);
    }

    return soma * h;
}

double trapezio_paralelo_critical(func_t fn, double a, double b, long n) {
    double h = (b - a) / (double) n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++) {
        double parcial = fn(a + i * h);

        #pragma omp critical
        {
            soma += parcial;
        }
    }

    return soma * h;
}

double trapezio_paralelo_atomic(func_t fn, double a, double b, long n) {
    double h = (b - a) / (double) n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++) {
        double parcial = fn(a + i * h);

        #pragma omp atomic
        soma += parcial;
    }

    return soma * h;
}

double trapezio_paralelo_reduction(func_t fn, double a, double b, long n) {
    double h = (b - a) / (double) n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for reduction(+:soma)
    for (long i = 1; i < n; i++) {
        soma += fn(a + i * h);
    }

    return soma * h;
}

typedef double (*trapezio_fn_t)(func_t, double, double, long);

typedef struct {
    const char *nome;
    double resultado;
    double tempo;
} resultado_t;

static resultado_t executar(const char *nome, trapezio_fn_t versao,
                             func_t fn, double a, double b, long n) {
    double inicio = omp_get_wtime();
    double resultado = versao(fn, a, b, n);
    double fim = omp_get_wtime();

    printf("Metodo: %-20s | Resultado: %.10f | Tempo: %.6fs\n",
           nome, resultado, fim - inicio);

    return (resultado_t){nome, resultado, fim - inicio};
}

static void exportar_csv(const char *caminho, resultado_t *resultados, int total) {
    FILE *arquivo = fopen(caminho, "w");
    if (!arquivo) {
        fprintf(stderr, "Erro ao criar arquivo CSV: %s\n", caminho);
        return;
    }

    fprintf(arquivo, "metodo,resultado,tempo\n");
    for (int i = 0; i < total; i++) {
        fprintf(arquivo, "%s,%.10f,%.6f\n",
                resultados[i].nome, resultados[i].resultado, resultados[i].tempo);
    }

    fclose(arquivo);
    printf("Resultados exportados para %s\n", caminho);
}

int main(int argc, char *argv[]) {
    double a = 0.0;
    double b = 1.0;
    long n = 100000000L;
    int num_threads = omp_get_max_threads();
    const char *csv_path = "resultados.csv";

    if (argc >= 2) a = atof(argv[1]);
    if (argc >= 3) b = atof(argv[2]);
    if (argc >= 4) n = atol(argv[3]);
    if (argc >= 5) num_threads = atoi(argv[4]);
    if (argc >= 6) csv_path = argv[5];

    if (n <= 0) {
        fprintf(stderr, "Erro: n deve ser um inteiro positivo.\n");
        return EXIT_FAILURE;
    }
    if (num_threads <= 0) {
        fprintf(stderr, "Erro: numero de threads deve ser positivo.\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(num_threads);

    resultado_t resultados[5];

    resultados[0] = executar("Sequencial", trapezio_sequencial, f, a, b, n);
    resultados[1] = executar("Sem protecao", trapezio_paralelo_sem_protecao, f, a, b, n);
    resultados[2] = executar("Critical", trapezio_paralelo_critical, f, a, b, n);
    resultados[3] = executar("Atomic", trapezio_paralelo_atomic, f, a, b, n);
    resultados[4] = executar("Reduction", trapezio_paralelo_reduction, f, a, b, n);

    exportar_csv(csv_path, resultados, 5);

    return EXIT_SUCCESS;
}