#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI_REF 3.14159265358979323846

typedef double (*func_t)(double);

static double f(double x)
{
    return 4.0 / (1.0 + x * x);
}

double trapezio_sequencial(func_t fn, double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double soma = (fn(a) + fn(b)) / 2.0;

    for (long i = 1; i < n; i++)
    {
        soma += fn(a + i * h);
    }

    return soma * h;
}

double trapezio_paralelo_sem_protecao(func_t fn, double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++)
    {
        soma += fn(a + i * h);
    }

    return soma * h;
}

double trapezio_paralelo_critical(func_t fn, double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++)
    {
        double parcial = fn(a + i * h);

        #pragma omp critical
        {
            soma += parcial;
        }
    }

    return soma * h;
}

double trapezio_paralelo_atomic(func_t fn, double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for
    for (long i = 1; i < n; i++)
    {
        double parcial = fn(a + i * h);

        #pragma omp atomic
        soma += parcial;
    }

    return soma * h;
}

double trapezio_paralelo_reduction(func_t fn, double a, double b, long n)
{
    double h = (b - a) / (double)n;
    double soma = (fn(a) + fn(b)) / 2.0;

    #pragma omp parallel for reduction(+:soma)
    for (long i = 1; i < n; i++)
    {
        soma += fn(a + i * h);
    }

    return soma * h;
}

typedef double (*trapezio_fn_t)(func_t, double, double, long);

typedef struct
{
    const char *nome;
    double resultado;
    double tempo;
} resultado_t;

static resultado_t executar(
    const char *nome,
    trapezio_fn_t versao,
    func_t fn,
    double a,
    double b,
    long n)
{
    double inicio = omp_get_wtime();

    double resultado = versao(fn, a, b, n);

    double fim = omp_get_wtime();

    double tempo = fim - inicio;

    printf(
        "N: %-10ld | Metodo: %-20s | Resultado: %.10f | Tempo: %.6fs\n",
        n,
        nome,
        resultado,
        tempo
    );

    return (resultado_t){nome, resultado, tempo};
}

static void exportar_csv(
    const char *caminho,
    long tamanho_n,
    resultado_t *resultados,
    int total)
{
    FILE *arquivo = fopen(caminho, "a");

    if (!arquivo)
    {
        fprintf(stderr, "Erro ao abrir arquivo CSV: %s\n", caminho);
        return;
    }

    for (int i = 0; i < total; i++)
    {
        fprintf(
            arquivo,
            "%ld,%s,%.10f,%.6f\n",
            tamanho_n,
            resultados[i].nome,
            resultados[i].resultado,
            resultados[i].tempo
        );
    }

    fclose(arquivo);
}

int main()
{
    double a = 0.0;
    double b = 1.0;

    const size_t valores_n[] =
    {
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000
    };

    const size_t quantidade_n =
        sizeof(valores_n) / sizeof(valores_n[0]);

    int num_threads = omp_get_max_threads();

    omp_set_num_threads(num_threads);

    const char *csv_path = "resultados.csv";

    FILE *arquivo = fopen(csv_path, "w");

    if (!arquivo)
    {
        fprintf(stderr, "Erro ao criar arquivo CSV: %s\n", csv_path);
        return EXIT_FAILURE;
    }

    fprintf(
        arquivo,
        "tamanho_n,metodo,resultado,tempo\n"
    );

    fclose(arquivo);

    for (size_t i = 0; i < quantidade_n; i++)
    {
        long n = valores_n[i];

        printf("\n============================================\n");
        printf("N = %ld\n", n);
        printf("============================================\n");

        resultado_t resultados[5];

        resultados[0] = executar(
            "Sequencial",
            trapezio_sequencial,
            f,
            a,
            b,
            n
        );

        resultados[1] = executar(
            "Sem protecao",
            trapezio_paralelo_sem_protecao,
            f,
            a,
            b,
            n
        );

        resultados[2] = executar(
            "Critical",
            trapezio_paralelo_critical,
            f,
            a,
            b,
            n
        );

        resultados[3] = executar(
            "Atomic",
            trapezio_paralelo_atomic,
            f,
            a,
            b,
            n
        );

        resultados[4] = executar(
            "Reduction",
            trapezio_paralelo_reduction,
            f,
            a,
            b,
            n
        );

        exportar_csv(
            csv_path,
            n,
            resultados,
            5
        );
    }

    printf("\nResultados exportados para %s\n", csv_path);

    return EXIT_SUCCESS;
}