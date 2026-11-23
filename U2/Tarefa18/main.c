#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 200
#define NZ 100

#define DX 10.0
#define DZ 10.0

#define DT 0.001
#define NT 1000

#define VELOCIDADE 2000.0

#define SZ 50
#define RZ 5

#define PI 3.14159265358979323846

#define LIDER 0

#define MAX_SISMOGRAMAS_SALVOS 16

#define TAG_TAREFA 1
#define TAG_RESULTADO 2
#define TAG_FIM 3

static const size_t quantidades_fontes[] = {1, 2, 4, 8, 16, 32, 64, 128};
static const size_t total_experimentos =
    sizeof(quantidades_fontes) / sizeof(quantidades_fontes[0]);

typedef struct {
    int id;
    int sx;
    int sz;
} Tiro;

typedef struct {
    int id;
    int trabalhador;
    double tempo_s;
    double amp_max;
} Resultado;

static MPI_Datatype cria_tipo_tiro(void)
{
    int blocos[3] = {1, 1, 1};
    MPI_Aint deslocamentos[3] = {offsetof(Tiro, id), offsetof(Tiro, sx),
                                 offsetof(Tiro, sz)};
    MPI_Datatype tipos[3] = {MPI_INT, MPI_INT, MPI_INT};

    MPI_Datatype tmp, tipo;
    MPI_Type_create_struct(3, blocos, deslocamentos, tipos, &tmp);

    MPI_Type_create_resized(tmp, 0, sizeof(Tiro), &tipo);
    MPI_Type_commit(&tipo);
    MPI_Type_free(&tmp);
    return tipo;
}

static MPI_Datatype cria_tipo_resultado(void)
{
    int blocos[4] = {1, 1, 1, 1};
    MPI_Aint deslocamentos[4] = {
        offsetof(Resultado, id), offsetof(Resultado, trabalhador),
        offsetof(Resultado, tempo_s), offsetof(Resultado, amp_max)};
    MPI_Datatype tipos[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Datatype tmp, tipo;
    MPI_Type_create_struct(4, blocos, deslocamentos, tipos, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Resultado), &tipo);
    MPI_Type_commit(&tipo);
    MPI_Type_free(&tmp);
    return tipo;
}

static double ricker(double t, double f0)
{
    double t0 = 1.0 / f0;
    double x = PI * f0 * (t - t0);
    return (1.0 - 2.0 * x * x) * exp(-x * x);
}

static void inicializa_modelo(double *vel)
{
    for (int iz = 0; iz < NZ; iz++)
        for (int ix = 0; ix < NX; ix++)
            vel[iz * NX + ix] = VELOCIDADE;
}

static double processa_tiro(const Tiro *tiro, double *vel, double *u,
                            double *u_old, double *u_new, float *sismograma)
{
    memset(u, 0, NX * NZ * sizeof(double));
    memset(u_old, 0, NX * NZ * sizeof(double));
    memset(u_new, 0, NX * NZ * sizeof(double));

    double f0 = 20.0;
    double dt2 = DT * DT;
    double amp_max = 0.0;
    int source_id = tiro->sz * NX + tiro->sx;

    for (int it = 0; it < NT; it++) {
        double t = it * DT;

        for (int iz = 1; iz < NZ - 1; iz++) {
            for (int ix = 1; ix < NX - 1; ix++) {
                int id = iz * NX + ix;

                double d2udx2 =
                    (u[id + 1] - 2.0 * u[id] + u[id - 1]) / (DX * DX);
                double d2udz2 =
                    (u[id + NX] - 2.0 * u[id] + u[id - NX]) / (DZ * DZ);

                u_new[id] = 2.0 * u[id] - u_old[id] +
                            vel[id] * vel[id] * dt2 * (d2udx2 + d2udz2);
            }
        }

        u_new[source_id] += ricker(t, f0);

        for (int ix = 0; ix < NX; ix++) {
            double a = u_new[RZ * NX + ix];
            sismograma[(size_t)it * NX + ix] = (float)a;
            if (fabs(a) > amp_max) amp_max = fabs(a);
        }

        double *tmp = u_old;
        u_old = u;
        u = u_new;
        u_new = tmp;
    }

    if (tiro->id >= MAX_SISMOGRAMAS_SALVOS) return amp_max;

    char nome[128];
    snprintf(nome, sizeof(nome), "outputs/sismograma_tiro_%03d.bin", tiro->id);
    FILE *fp = fopen(nome, "wb");
    if (fp != NULL) {
        fwrite(sismograma, sizeof(float), (size_t)NT * NX, fp);
        fclose(fp);
    }

    return amp_max;
}

static void trabalhador(int rank, MPI_Datatype tipo_tiro,
                        MPI_Datatype tipo_resultado)
{
    double *vel = malloc(NX * NZ * sizeof(double));
    double *u = malloc(NX * NZ * sizeof(double));
    double *u_old = malloc(NX * NZ * sizeof(double));
    double *u_new = malloc(NX * NZ * sizeof(double));
    float *sismograma = malloc((size_t)NT * NX * sizeof(float));

    if (!vel || !u || !u_old || !u_new || !sismograma) {
        fprintf(stderr, "[%d] Erro de alocacao de memoria\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    inicializa_modelo(vel);

    while (1) {
        Tiro tiro;
        MPI_Status status;

        MPI_Recv(&tiro, 1, tipo_tiro, LIDER, MPI_ANY_TAG, MPI_COMM_WORLD,
                 &status);

        if (status.MPI_TAG == TAG_FIM) break;

        double ini = MPI_Wtime();
        double amp = processa_tiro(&tiro, vel, u, u_old, u_new, sismograma);
        double fim = MPI_Wtime();

        Resultado r = {tiro.id, rank, fim - ini, amp};
        MPI_Send(&r, 1, tipo_resultado, LIDER, TAG_RESULTADO, MPI_COMM_WORLD);
    }

    free(vel);
    free(u);
    free(u_old);
    free(u_new);
    free(sismograma);
}

typedef struct {
    int trabalhador;
    double inicio_s;
    double fim_s;
    double tempo_s;
    double amp_max;
} RegistroTiro;

static Tiro monta_tiro(int id, int n_fontes)
{
    Tiro t;
    t.id = id;

    t.sx = n_fontes == 1 ? NX / 2 : 10 + id * (NX - 21) / (n_fontes - 1);
    t.sz = SZ;
    return t;
}

static void lider_experimento(int n_fontes, int processos,
                              MPI_Datatype tipo_tiro,
                              MPI_Datatype tipo_resultado, FILE *json,
                              int ultimo)
{
    int trabalhadores = processos - 1;
    RegistroTiro *registros = calloc(n_fontes, sizeof(RegistroTiro));
    int *tarefas = calloc(processos, sizeof(int));
    double *ocupado = calloc(processos, sizeof(double));

    int proximo = 0;
    int ativos = 0;

    double t0 = MPI_Wtime();

    for (int w = 1; w <= trabalhadores && proximo < n_fontes; w++) {
        Tiro tiro = monta_tiro(proximo, n_fontes);
        registros[proximo].trabalhador = w;
        registros[proximo].inicio_s = MPI_Wtime() - t0;
        MPI_Send(&tiro, 1, tipo_tiro, w, TAG_TAREFA, MPI_COMM_WORLD);
        proximo++;
        ativos++;
    }

    while (ativos > 0) {
        Resultado r;
        MPI_Status status;
        MPI_Recv(&r, 1, tipo_resultado, MPI_ANY_SOURCE, TAG_RESULTADO,
                 MPI_COMM_WORLD, &status);

        int w = status.MPI_SOURCE;
        registros[r.id].fim_s = MPI_Wtime() - t0;
        registros[r.id].tempo_s = r.tempo_s;
        registros[r.id].amp_max = r.amp_max;
        tarefas[w]++;
        ocupado[w] += r.tempo_s;

        if (proximo < n_fontes) {
            Tiro tiro = monta_tiro(proximo, n_fontes);
            registros[proximo].trabalhador = w;
            registros[proximo].inicio_s = MPI_Wtime() - t0;
            MPI_Send(&tiro, 1, tipo_tiro, w, TAG_TAREFA, MPI_COMM_WORLD);
            proximo++;
        } else {
            ativos--;
        }
    }

    double tempo_total = MPI_Wtime() - t0;

    double soma = 0.0;
    for (int i = 0; i < n_fontes; i++) soma += registros[i].tempo_s;

    double speedup = soma / tempo_total;
    double eficiencia = speedup / trabalhadores;

    printf("Fontes: %3d | tempo total: %8.4f s | soma dos tiros: %8.4f s | "
           "speedup: %5.2f | eficiencia: %5.1f%% | tarefas/trab:",
           n_fontes, tempo_total, soma, speedup, 100.0 * eficiencia);
    for (int w = 1; w <= trabalhadores; w++) printf(" %d", tarefas[w]);
    printf("\n");

    fprintf(json, "    {\n");
    fprintf(json, "      \"n_fontes\": %d,\n", n_fontes);
    fprintf(json, "      \"tempo_total_s\": %f,\n", tempo_total);
    fprintf(json, "      \"soma_tempo_tiros_s\": %f,\n", soma);
    fprintf(json, "      \"tempo_medio_tiro_s\": %f,\n", soma / n_fontes);
    fprintf(json, "      \"speedup\": %f,\n", speedup);
    fprintf(json, "      \"eficiencia\": %f,\n", eficiencia);

    fprintf(json, "      \"tarefas_por_trabalhador\": [");
    for (int w = 1; w <= trabalhadores; w++)
        fprintf(json, "%d%s", tarefas[w], w < trabalhadores ? ", " : "");
    fprintf(json, "],\n");

    fprintf(json, "      \"tempo_ocupado_por_trabalhador_s\": [");
    for (int w = 1; w <= trabalhadores; w++)
        fprintf(json, "%f%s", ocupado[w], w < trabalhadores ? ", " : "");
    fprintf(json, "],\n");

    fprintf(json, "      \"tiros\": [\n");
    for (int i = 0; i < n_fontes; i++) {
        Tiro t = monta_tiro(i, n_fontes);
        fprintf(json,
                "        {\"id\": %d, \"sx\": %d, \"sz\": %d, "
                "\"trabalhador\": %d, \"inicio_s\": %f, \"fim_s\": %f, "
                "\"tempo_s\": %f, \"amp_max\": %g}%s\n",
                i, t.sx, t.sz, registros[i].trabalhador,
                registros[i].inicio_s, registros[i].fim_s,
                registros[i].tempo_s, registros[i].amp_max,
                i + 1 < n_fontes ? "," : "");
    }
    fprintf(json, "      ]\n");
    fprintf(json, "    }%s\n", ultimo ? "" : ",");

    free(registros);
    free(tarefas);
    free(ocupado);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processos);

    if (processos < 3) {
        if (rank == LIDER)
            fprintf(stderr, "Use pelo menos 3 processos "
                            "(1 lider + 2 trabalhadores).\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Datatype tipo_tiro = cria_tipo_tiro();
    MPI_Datatype tipo_resultado = cria_tipo_resultado();

    if (rank == LIDER) {
        char nome[64];
        snprintf(nome, sizeof(nome), "resultados_np%d.json", processos);
        FILE *json = fopen(nome, "w");
        if (json == NULL) {
            fprintf(stderr, "Erro ao abrir %s\n", nome);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Lider-Trabalhador: %d processos (1 lider + %d trabalhadores)\n",
               processos, processos - 1);
        printf("Malha %dx%d, %d passos de tempo por tiro\n\n", NX, NZ, NT);

        fprintf(json, "{\n");
        fprintf(json, "  \"processos\": %d,\n", processos);
        fprintf(json, "  \"trabalhadores\": %d,\n", processos - 1);
        fprintf(json, "  \"nx\": %d, \"nz\": %d, \"nt\": %d,\n", NX, NZ, NT);
        fprintf(json, "  \"experimentos\": [\n");

        for (size_t e = 0; e < total_experimentos; e++) {
            lider_experimento((int)quantidades_fontes[e], processos, tipo_tiro,
                              tipo_resultado, json,
                              e + 1 == total_experimentos);
        }

        fprintf(json, "  ]\n}\n");
        fclose(json);

        Tiro vazio = {-1, 0, 0};
        for (int w = 1; w < processos; w++)
            MPI_Send(&vazio, 1, tipo_tiro, w, TAG_FIM, MPI_COMM_WORLD);

        printf("\nResultados salvos em %s\n", nome);
    } else {
        trabalhador(rank, tipo_tiro, tipo_resultado);
    }

    MPI_Type_free(&tipo_tiro);
    MPI_Type_free(&tipo_resultado);

    MPI_Finalize();
    return 0;
}
