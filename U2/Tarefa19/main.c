#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAIZ 0
#define REPETICOES 5

static const size_t tamanhos[] = {1000, 2000, 4000, 8000, 12000, 16000};
static const size_t quantidade_tamanhos =
    sizeof(tamanhos) / sizeof(tamanhos[0]);

static inline double valor_a(size_t i, size_t j)
{
    return (double)((i * 7 + j * 3) % 17) * 0.125;
}

static inline double valor_x(size_t j) { return 1.0 / (double)(j % 100 + 1); }

static void matvec(const double *A, const double *x, double *y, size_t linhas,
                   size_t n)
{
    for (size_t i = 0; i < linhas; i++) {
        const double *linha = A + i * n;
        double soma = 0.0;
        for (size_t j = 0; j < n; j++) soma += linha[j] * x[j];
        y[i] = soma;
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processos);

    size_t limite = argc > 1 ? (size_t)atol(argv[1]) : (size_t)-1;

    char no[MPI_MAX_PROCESSOR_NAME];
    int tam_nome;
    memset(no, 0, sizeof(no));
    MPI_Get_processor_name(no, &tam_nome);
    char *nomes = NULL;
    if (rank == RAIZ) nomes = malloc((size_t)processos * MPI_MAX_PROCESSOR_NAME);
    MPI_Gather(no, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, nomes,
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR, RAIZ, MPI_COMM_WORLD);

    FILE *json = NULL;
    if (rank == RAIZ) {
        int nos = 0;
        for (int p = 0; p < processos; p++) {
            int novo = 1;
            for (int q = 0; q < p; q++)
                if (strcmp(nomes + p * MPI_MAX_PROCESSOR_NAME,
                           nomes + q * MPI_MAX_PROCESSOR_NAME) == 0) {
                    novo = 0;
                    break;
                }
            nos += novo;
        }

        char nome_arquivo[64];
        snprintf(nome_arquivo, sizeof(nome_arquivo), "resultados_np%d.json",
                 processos);
        json = fopen(nome_arquivo, "w");
        if (json == NULL) {
            fprintf(stderr, "Erro ao abrir %s\n", nome_arquivo);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(json, "{\n  \"processos\": %d,\n  \"nos\": %d,\n", processos,
                nos);
        fprintf(json, "  \"repeticoes\": %d,\n  \"resultados\": [\n",
                REPETICOES);

        printf("y = A.x com %d processos em %d no(s)\n", processos, nos);
        printf("%7s | %10s %10s %10s %10s %10s | %10s %8s %8s | %s\n", "N",
               "scatter", "bcast", "calculo", "gather", "total", "seq",
               "speedup", "efic.", "erro");
    }

    int primeiro = 1;

    for (size_t t = 0; t < quantidade_tamanhos; t++) {
        size_t m = tamanhos[t];
        size_t n = tamanhos[t];
        if (m > limite) break;

        size_t linhas_local = (m + processos - 1) / processos;
        size_t m_pad = linhas_local * processos;

        double *A = NULL, *y = NULL, *y_seq = NULL;
        double *x = malloc(n * sizeof(double));
        double *A_local = malloc(linhas_local * n * sizeof(double));
        double *y_local = malloc(linhas_local * sizeof(double));

        if (rank == RAIZ) {
            A = calloc(m_pad * n, sizeof(double));
            y = malloc(m_pad * sizeof(double));
            y_seq = malloc(m * sizeof(double));
            if (!A || !y || !y_seq) {
                fprintf(stderr, "Erro de alocacao (N = %zu)\n", n);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (size_t i = 0; i < m; i++)
                for (size_t j = 0; j < n; j++) A[i * n + j] = valor_a(i, j);
            for (size_t j = 0; j < n; j++) x[j] = valor_x(j);
        }
        if (!x || !A_local || !y_local) {
            fprintf(stderr, "Erro de alocacao (N = %zu)\n", n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double tempo_seq = 0.0;
        if (rank == RAIZ) {
            for (int r = 0; r < REPETICOES; r++) {
                double t0 = MPI_Wtime();
                matvec(A, x, y_seq, m, n);
                tempo_seq += MPI_Wtime() - t0;
            }
            tempo_seq /= REPETICOES;
        }

        int contagem = (int)(linhas_local * n);
        double soma_fases[5] = {0, 0, 0, 0, 0};

        for (int r = 0; r < REPETICOES; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            MPI_Scatter(A, contagem, MPI_DOUBLE, A_local, contagem,
                        MPI_DOUBLE, RAIZ, MPI_COMM_WORLD);
            double t1 = MPI_Wtime();

            MPI_Bcast(x, (int)n, MPI_DOUBLE, RAIZ, MPI_COMM_WORLD);
            double t2 = MPI_Wtime();

            matvec(A_local, x, y_local, linhas_local, n);
            double t3 = MPI_Wtime();

            MPI_Gather(y_local, (int)linhas_local, MPI_DOUBLE, y,
                       (int)linhas_local, MPI_DOUBLE, RAIZ, MPI_COMM_WORLD);
            double t4 = MPI_Wtime();

            double locais[5] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0};
            double maximos[5];
            MPI_Reduce(locais, maximos, 5, MPI_DOUBLE, MPI_MAX, RAIZ,
                       MPI_COMM_WORLD);
            if (rank == RAIZ)
                for (int f = 0; f < 5; f++) soma_fases[f] += maximos[f];
        }

        if (rank == RAIZ) {
            double fases[5];
            for (int f = 0; f < 5; f++) fases[f] = soma_fases[f] / REPETICOES;

            double erro = 0.0;
            for (size_t i = 0; i < m; i++) {
                double d = fabs(y[i] - y_seq[i]);
                if (d > erro) erro = d;
            }

            double speedup = tempo_seq / fases[4];
            double eficiencia = speedup / processos;
            double speedup_calculo = tempo_seq / fases[2];

            printf("%7zu | %10.5f %10.5f %10.5f %10.5f %10.5f | %10.5f %8.2f "
                   "%7.1f%% | %.1e\n",
                   n, fases[0], fases[1], fases[2], fases[3], fases[4],
                   tempo_seq, speedup, 100 * eficiencia, erro);

            fprintf(json,
                    "%s    {\"m\": %zu, \"n\": %zu, \"linhas_por_processo\": "
                    "%zu, \"tempo_scatter_s\": %.9f, \"tempo_bcast_s\": %.9f, "
                    "\"tempo_calculo_s\": %.9f, \"tempo_gather_s\": %.9f, "
                    "\"tempo_total_s\": %.9f, \"tempo_sequencial_s\": %.9f, "
                    "\"speedup\": %f, \"eficiencia\": %f, "
                    "\"speedup_calculo\": %f, \"erro_max\": %g}",
                    primeiro ? "" : ",\n", m, n, linhas_local, fases[0],
                    fases[1], fases[2], fases[3], fases[4], tempo_seq,
                    speedup, eficiencia, speedup_calculo, erro);
            primeiro = 0;
        }

        free(A);
        free(y);
        free(y_seq);
        free(x);
        free(A_local);
        free(y_local);
    }

    if (rank == RAIZ) {
        fprintf(json, "\n  ]\n}\n");
        fclose(json);
        free(nomes);
        printf("Resultados salvos em resultados_np%d.json\n", processos);
    }

    MPI_Finalize();
    return 0;
}
