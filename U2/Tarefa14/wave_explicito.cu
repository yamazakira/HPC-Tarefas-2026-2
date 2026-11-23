#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int NX = 200;
static int NZ = 100;

static const size_t tamanhos[] = {200, 400, 800, 1600, 3200, 6400, 12800};
#define NUM_TAMANHOS (sizeof(tamanhos) / sizeof(tamanhos[0]))

static int salvar_campo = 0;
static double soma_receptor = 0.0;
static double tempo_h2d = 0.0;
static double tempo_d2h = 0.0;

static double agora(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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
    for (int iz = 0; iz < NZ; iz++) {
        for (int ix = 0; ix < NX; ix++) {

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

    if (fp == NULL) {
        perror("Erro ao abrir arquivo");
        exit(EXIT_FAILURE);
    }

    fwrite(
        campo,
        sizeof(double),
        NX * NZ,
        fp
    );

    fclose(fp);
}


__global__ void atualiza_campo(
    const double *vel,
    const double *u,
    const double *u_old,
    double *u_new,
    double dt2,
    int NX,
    int NZ)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= 1 && ix < NX - 1 && iz >= 1 && iz < NZ - 1) {

        int id = iz * NX + ix;

        double d2udx2 =
            (u[id + 1]
           - 2.0 * u[id]
           + u[id - 1])
            / (DX * DX);


        double d2udz2 =
            (u[id + NX]
           - 2.0 * u[id]
           + u[id - NX])
            / (DZ * DZ);


        u_new[id] =
            2.0 * u[id]
            - u_old[id]
            + vel[id] * vel[id]
            * dt2
            * (d2udx2 + d2udz2);
    }
}


__global__ void injeta_fonte(double *u_new, int source_id, double valor)
{
    u_new[source_id] += valor;
}


/* --------------------------------------------------
 * Propagação da onda
 * -------------------------------------------------- */
static double simula(void)
{
    /*
     * Campo de velocidade
     */
    double *vel = NULL;

    /*
     * Campo de onda atual
     */
    double *u = NULL;

    /*
     * Campo de onda anterior
     */
    double *u_old = NULL;

    /*
     * Campo de onda futuro
     */
    double *u_new = NULL;

    vel = (double *)malloc(NX * NZ * sizeof(double));
    u = (double *)calloc(NX * NZ, sizeof(double));
    u_old = (double *)calloc(NX * NZ, sizeof(double));
    u_new = (double *)calloc(NX * NZ, sizeof(double));

    double *d_vel = NULL, *d_u = NULL, *d_u_old = NULL, *d_u_new = NULL;

    cudaMalloc(&d_vel, NX * NZ * sizeof(double));
    cudaMalloc(&d_u, NX * NZ * sizeof(double));
    cudaMalloc(&d_u_old, NX * NZ * sizeof(double));
    cudaMalloc(&d_u_new, NX * NZ * sizeof(double));

    if (!vel || !u || !u_old || !u_new || !d_vel || !d_u || !d_u_old || !d_u_new) {
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

    if (receiver == NULL) {
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


    dim3 block(16, 16);
    dim3 grid((NX + block.x - 1) / block.x, (NZ + block.y - 1) / block.y);


    soma_receptor = 0.0;
    double inicio = agora();

    cudaMemcpy(d_vel, vel, NX * NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, NX * NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_old, u_old, NX * NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, u_new, NX * NZ * sizeof(double), cudaMemcpyHostToDevice);

    tempo_h2d = agora() - inicio;

    /* ------------------------------------------------
     * Loop temporal
     * ------------------------------------------------ */
    for (int it = 0; it < NT; it++) {

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

        atualiza_campo<<<grid, block>>>(d_vel, d_u, d_u_old, d_u_new, dt2, NX, NZ);

        cudaDeviceSynchronize();


        /*
         * ---------------------------------------------
         * Injeta a fonte
         * ---------------------------------------------
         */

        int source_id = SZ * NX + SX;

        injeta_fonte<<<1, 1>>>(d_u_new, source_id, ricker(t, f0));


        /*
         * ---------------------------------------------
         * Registra o receptor
         * ---------------------------------------------
         */

        int receiver_id = RZ * NX + RX;

        double amostra;
        cudaMemcpy(&amostra, d_u_new + receiver_id, sizeof(double), cudaMemcpyDeviceToHost);

        fprintf(
            receiver,
            "%lf %lf\n",
            t,
            amostra
        );

        soma_receptor += fabs(amostra);


        /*
         * ---------------------------------------------
         * Salva o campo de onda
         * ---------------------------------------------
         */

        char nome[100];

        sprintf(
            nome,
            "wavefield_%04d.bin",
            it
        );

        if (salvar_campo) {
            cudaMemcpy(u_new, d_u_new, NX * NZ * sizeof(double), cudaMemcpyDeviceToHost);
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

        tmp = d_u_old;
        d_u_old = d_u;
        d_u = d_u_new;
        d_u_new = tmp;


        if (it % 100 == 0) {
            printf(
                "Passo %d / %d\n",
                it,
                NT
            );
        }
    }


    double inicio_d2h = agora();

    cudaMemcpy(u, d_u, NX * NZ * sizeof(double), cudaMemcpyDeviceToHost);

    tempo_d2h = agora() - inicio_d2h;

    double tempo = agora() - inicio;

    fclose(receiver);

    cudaFree(d_vel);
    cudaFree(d_u);
    cudaFree(d_u_old);
    cudaFree(d_u_new);
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

    cudaFree(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    FILE *saida = fopen("resultados_explicita.json", "w");

    if (saida == NULL)
        return 1;

    fprintf(saida, "{\n  \"versao\": \"memoria_explicita\",\n  \"gpu\": \"%s\",\n"
                   "  \"nt\": %d,\n  \"resultados\": [\n", prop.name, NT);

    for (size_t i = 0; i < NUM_TAMANHOS; i++) {
        NX = (int)tamanhos[i];
        NZ = (int)tamanhos[i] / 2;

        double tempo = simula();

        printf("[EXPLICITA] %dx%d: %.6f s | H2D %.6f s | D2H %.6f s\n",
               NX, NZ, tempo, tempo_h2d, tempo_d2h);

        fprintf(saida,
                "    {\"nx\": %d, \"nz\": %d, \"tempo_total_s\": %.6f, "
                "\"tempo_h2d_s\": %.6f, \"tempo_d2h_s\": %.6f, "
                "\"soma_receptor\": %.10e}%s\n",
                NX, NZ, tempo, tempo_h2d, tempo_d2h, soma_receptor,
                i + 1 < NUM_TAMANHOS ? "," : "");
    }

    fprintf(saida, "  ]\n}\n");
    fclose(saida);

    return 0;
}
