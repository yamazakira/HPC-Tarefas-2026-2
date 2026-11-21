#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#pragma omp declare target
static int NX = 200;
static int NZ = 100;
#pragma omp end declare target

static const size_t tamanhos[] = {200, 400, 800, 1600, 3200};
#define NUM_TAMANHOS (sizeof(tamanhos) / sizeof(tamanhos[0]))

static int salvar_campo = 0;
static double soma_receptor = 0.0;
static long limiar_gpu = 250000;
static double tempo_h2d = 0.0;
static double tempo_d2h = 0.0;

#define DX 10.0
#define DZ 10.0

#define DT 0.001
#define NT 1000

#define VELOCIDADE 2000.0

/* Posicao da fonte */
#define SX 100
#define SZ 50

/* Posicao do receptor */
#define RX 150
#define RZ 50

#define PI 3.14159265358979323846

/* --------------------------------------------------
 * Wavelet Ricker
 * -------------------------------------------------- */
double ricker(double t, double f0)
{
    double t0 = 1.0 / f0;

    double x = PI * f0 * (t - t0);

    return (1.0 - 2.0 * x * x) * exp(-x * x);
}

/* --------------------------------------------------
 * Inicializa modelo de velocidade
 * -------------------------------------------------- */
void inicializa_modelo(double *vel)
{
    for (int iz = 0; iz < NZ; iz++)
    {
        for (int ix = 0; ix < NX; ix++)
        {

            /*
             * Modelo homogêneo.
             */
            vel[iz * NX + ix] = VELOCIDADE;
        }
    }
}

/* --------------------------------------------------
 * Salva campo de onda em binario
 * -------------------------------------------------- */
void salva_binario(
    const char *nome,
    double *campo)
{
    FILE *fp = fopen(nome, "wb");

    if (fp == NULL)
    {
        perror("Erro ao abrir arquivo");
        exit(EXIT_FAILURE);
    }

    fwrite(
        campo,
        sizeof(double),
        NX * NZ,
        fp);

    fclose(fp);
}

/* --------------------------------------------------
 * Propagação da onda
 * -------------------------------------------------- */
static double simula(void)
{
    /*
     * Campo de velocidade
     */
    double *vel = malloc(NX * NZ * sizeof(double));

    /*
     * Campo de onda atual
     */
    double *u = calloc(NX * NZ, sizeof(double));

    /*
     * Campo de onda anterior
     */
    double *u_old = calloc(NX * NZ, sizeof(double));

    /*
     * Campo de onda futuro
     */
    double *u_new = calloc(NX * NZ, sizeof(double));

    if (!vel || !u || !u_old || !u_new)
    {
        printf("Erro de alocacao de memoria\n");
        exit(EXIT_FAILURE);
    }

    /* Modelo de velocidade */
    inicializa_modelo(vel);

    /*
     * Arquivo para o sismograma
     *
     * Cada linha:
     *
     * tempo amplitude
     */
    FILE *receiver = fopen("receiver.dat", "w");

    if (receiver == NULL)
    {
        printf("Erro ao abrir receiver.dat\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Frequencia dominante da fonte
     */
    double f0 = 20.0;

    /*
     * Fator da equacao de diferencas finitas
     */
    double dt2 = DT * DT;

    soma_receptor = 0.0;
    int usa_gpu = (long)NX * NZ >= limiar_gpu;

    double inicio = omp_get_wtime();

#pragma omp target enter data if(usa_gpu)                         \
    map(to : vel[0 : NX * NZ], u[0 : NX * NZ], u_old[0 : NX * NZ], \
            u_new[0 : NX * NZ])

    tempo_h2d = omp_get_wtime() - inicio;

    /* ------------------------------------------------
     * Loop temporal
     * ------------------------------------------------ */
    for (int it = 0; it < NT; it++)
    {

        double t = it * DT;

        /*
         * ---------------------------------------------
         * Atualizacao do campo de onda
         *
         * d2u/dt2 =
         *
         * v^2 * (
         *     d2u/dx2 +
         *     d2u/dz2
         * )
         * ---------------------------------------------
         */

#pragma omp target teams distribute parallel for collapse(2) if(usa_gpu)
        for (int iz = 1; iz < NZ - 1; iz++)
        {

            for (int ix = 1; ix < NX - 1; ix++)
            {

                int id = iz * NX + ix;

                double d2udx2 =
                    (u[id + 1] - 2.0 * u[id] + u[id - 1]) / (DX * DX);

                double d2udz2 =
                    (u[id + NX] - 2.0 * u[id] + u[id - NX]) / (DZ * DZ);

                u_new[id] =
                    2.0 * u[id] - u_old[id] + vel[id] * vel[id] * dt2 * (d2udx2 + d2udz2);
            }
        }

        /*
         * ---------------------------------------------
         * Injeta a fonte
         * ---------------------------------------------
         */

        int source_id = SZ * NX + SX;

        double fonte = ricker(t, f0);

#pragma omp target if(usa_gpu)
        u_new[source_id] += fonte;

        /*
         * ---------------------------------------------
         * Registra o receptor
         * ---------------------------------------------
         */

        int receiver_id = RZ * NX + RX;

#pragma omp target update if(usa_gpu) from(u_new[receiver_id : 1])

        fprintf(
            receiver,
            "%lf %lf\n",
            t,
            u_new[receiver_id]);

        soma_receptor += fabs(u_new[receiver_id]);

        /*
         * ---------------------------------------------
         * Salva o campo de onda
         * ---------------------------------------------
         */

        char nome[100];

        sprintf(
            nome,
            "wavefield_%04d.bin",
            it);

        if (salvar_campo)
        {
#pragma omp target update if(usa_gpu) from(u_new[0 : NX * NZ])
            salva_binario(nome, u_new);
        }

        /*
         * ---------------------------------------------
         * Troca dos campos
         *
         * old <- atual
         * atual <- novo
         * novo <- antigo
         * ---------------------------------------------
         */

        double *tmp;

        tmp = u_old;
        u_old = u;

        u = u_new;

        u_new = tmp;

        if (it % 100 == 0)
        {
            printf(
                "Passo %d / %d\n",
                it,
                NT);
        }
    }

    double inicio_d2h = omp_get_wtime();

#pragma omp target exit data if(usa_gpu) map(from : u[0 : NX * NZ]) \
    map(release : vel[0 : NX * NZ], u_old[0 : NX * NZ], u_new[0 : NX * NZ])

    tempo_d2h = omp_get_wtime() - inicio_d2h;

    double tempo = omp_get_wtime() - inicio;

    fclose(receiver);

    free(vel);
    free(u);
    free(u_old);
    free(u_new);

    printf("\nSimulacao finalizada.\n");

    return tempo;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
        salvar_campo = atoi(argv[1]);

    const char *env_limiar = getenv("LIMIAR_GPU");
    long limiar_auto = env_limiar ? atol(env_limiar) : limiar_gpu;

#pragma omp target
    {
    }

    FILE *saida = fopen("resultados_otimizado.json", "w");

    if (saida == NULL)
        return 1;

    fprintf(saida, "{\n  \"versao\": \"otimizado\",\n  \"dispositivos\": %d,\n"
                   "  \"threads\": %d,\n  \"nt\": %d,\n  \"limiar_gpu\": %ld,\n"
                   "  \"resultados\": [\n",
            omp_get_num_devices(), omp_get_max_threads(), NT, limiar_auto);

    const char *modos[] = {"auto", "gpu", "cpu"};
    const long limiares[] = {limiar_auto, 0, -1};
    int primeiro = 1;

    for (size_t i = 0; i < NUM_TAMANHOS; i++)
    {
        NX = (int)tamanhos[i];
        NZ = (int)tamanhos[i] / 2;
#pragma omp target update to(NX, NZ)

        for (int m = 0; m < 3; m++)
        {
            limiar_gpu = limiares[m] < 0 ? (long)NX * NZ + 1 : limiares[m];
            int usa_gpu = (long)NX * NZ >= limiar_gpu;

            double tempo = simula();

            printf("[OTIMIZADO %s] %dx%d (%s): %.6f s | H2D %.6f s | D2H %.6f s\n",
                   modos[m], NX, NZ, usa_gpu ? "gpu" : "cpu", tempo, tempo_h2d, tempo_d2h);

            fprintf(saida,
                    "%s    {\"nx\": %d, \"nz\": %d, \"modo\": \"%s\", \"executou_em\": \"%s\", "
                    "\"tempo_total_s\": %.6f, \"tempo_h2d_s\": %.6f, \"tempo_d2h_s\": %.6f, "
                    "\"soma_receptor\": %.10e}",
                    primeiro ? "" : ",\n", NX, NZ, modos[m], usa_gpu ? "gpu" : "cpu",
                    tempo, tempo_h2d, tempo_d2h, soma_receptor);
            primeiro = 0;
        }
    }

    fprintf(saida, "\n  ]\n}\n");
    fclose(saida);

    return 0;
}
