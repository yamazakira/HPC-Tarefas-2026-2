#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI_REF 3.14159265358979323846

typedef double (*func_t)(double);

/* Funcao de exemplo cujo comportamento sera vetorizado */
static double f(double x)
{
    return sin(x);
}

int main(void)
{
    const double a = 0.0;
    const double b = 2.0 * PI_REF;
    const size_t n = 1000000UL;
    const int threads_sections = 2;
    const int threads_internas = 4;

    const double h = (b - a) / (double)(n - 1);

    double *vetor = malloc(n * sizeof(double));
    double *derivada = malloc(n * sizeof(double));

    if (!vetor || !derivada)
    {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < n; i++)
    {
        vetor[i] = f(a + (double)i * h);
    }

    omp_set_nested(1);
    omp_set_max_active_levels(2);

    double resultado_integral = 0.0;

    #pragma omp parallel sections num_threads(threads_sections)
    {
        #pragma omp section
        {
            double soma = (vetor[0] + vetor[n - 1]) / 2.0;
            
            #pragma omp parallel num_threads(threads_internas)
            {
                #pragma omp for reduction(+ : soma)
                
                for (size_t i = 1; i < n - 1; i++)
                {
                    soma += vetor[i];
                }

                #pragma omp single
                {
                    resultado_integral = soma * h;
                    printf("[Integral - trapezio] resultado = %.10f "
                           "(impresso pela thread %d de %d, secao 1)\n",
                           resultado_integral, omp_get_thread_num(),
                           omp_get_num_threads());
                }
            }
        }
        #pragma omp section
        {
            #pragma omp parallel num_threads(threads_internas)
            {
                #pragma omp for
                for (size_t i = 1; i < n - 1; i++)
                {
                    derivada[i] = (vetor[i + 1] - vetor[i - 1]) / (2.0 * h);
                }
                
                #pragma omp single
                {
                    derivada[0] = (vetor[1] - vetor[0]) / h;
                    derivada[n - 1] = (vetor[n - 1] - vetor[n - 2]) / h;

                    printf("[Derivada - dif. finitas] concluida "
                           "(impresso pela thread %d de %d, secao 2). "
                           "Ex.: derivada[0]=%.6f  derivada[n/2]=%.6f\n",
                           omp_get_thread_num(), omp_get_num_threads(),
                           derivada[0], derivada[n / 2]);
                }
            }
        }
    }

    printf("\nProcessamento concluido. n = %zu amostras.\n", n);

    free(vetor);
    free(derivada);
    return EXIT_SUCCESS;
}