#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 200
#define NZ 100

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
int main()
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
        return 1;
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
        return 1;
    }

    /*
     * Frequencia dominante da fonte
     */
    double f0 = 20.0;

    /*
     * Fator da equacao de diferencas finitas
     */
    double dt2 = DT * DT;

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

#pragma omp parallel for collapse(2) /* <<< ALTERACAO (unica) */
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

        u_new[source_id] += ricker(t, f0);

        /*
         * ---------------------------------------------
         * Registra o receptor
         * ---------------------------------------------
         */

        int receiver_id = RZ * NX + RX;

        fprintf(
            receiver,
            "%lf %lf\n",
            t,
            u_new[receiver_id]);

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

        salva_binario(nome, u_new);

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

    fclose(receiver);

    free(vel);
    free(u);
    free(u_old);
    free(u_new);

    printf("\nSimulacao finalizada.\n");

    return 0;
}