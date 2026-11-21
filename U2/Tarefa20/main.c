/*
 * Tarefa 20 - Integral pelo metodo dos trapezios (MPI + OpenMP + GPU)
 *             com envio periodico da soma parcial ao processo vizinho
 *
 *   I = integral de 0 a 1 de 4 / (1 + x^2) dx = pi
 *
 * 1. Distribuicao: os N trapezios sao divididos em blocos contiguos entre
 *    os processos MPI.
 * 2. Paralelismo local com OpenMP. Se houver GPU disponivel (detectada em
 *    tempo de execucao com omp_get_num_devices / omp_is_initial_device),
 *    a parcela do processo e calculada na GPU com "omp target"; caso
 *    contrario, nas threads da CPU.
 * 3. Tolerancia a falhas: a cada 100 amostras processadas, a soma parcial
 *    e enviada ao vizinho (rank + 1) % P, que a guarda em uma variavel
 *    local (backup). Somente a thread 0 comunica (MPI_THREAD_FUNNELED);
 *    as demais threads continuam calculando.
 * 4. Reducao global com MPI_Reduce.
 * 5. Validacao contra o valor analitico (pi).
 *
 * Modos executados para cada N:
 *   cpu             - threads OpenMP na CPU, com envio ao vizinho
 *   cpu_sem_backup  - igual, mas sem envio (mede o custo da comunicacao)
 *   gpu             - calculo na GPU (so se houver dispositivo), com envio
 *
 * compilar (CPU):  mpicc -O2 -fopenmp main.c -o main -lm
 * compilar (GPU):  mpicc -O2 -fopenmp -foffload=nvptx-none -fno-stack-protector main.c -o main -lm
 *                  (ou NVHPC: mpicc -O2 -mp=gpu main.c -o main)
 * executar:        OMP_NUM_THREADS=8 mpirun -np 4 ./main [rotulo]
 *                  (gera resultados_np<P>.json ou resultados_<rotulo>_np<P>.json)
 */
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RAIZ 0
#define AMOSTRAS_CHECKPOINT 100 /* envia ao vizinho a cada 100 amostras */
#define BLOCOS_POR_PEGADA 1000  /* trabalho pego por thread: 1000 x 100 */
#define LOTES_GPU 64            /* parcela do processo dividida em lotes */
#define TAG_BACKUP 77
#define PAUSA_COMUNICACAO_NS 20000 /* intervalo entre verificacoes (20 us) */

#define A_INTERVALO 0.0
#define B_INTERVALO 1.0
#define VALOR_EXATO 3.14159265358979323846

/* Numeros de trapezios testados */
static const size_t quantidades[] = {1000000, 10000000, 100000000,
                                     1000000000};
static const size_t total_quantidades =
    sizeof(quantidades) / sizeof(quantidades[0]);

#pragma omp declare target
static inline double f(double x) { return 4.0 / (1.0 + x * x); }
#pragma omp end declare target

/* Slot de progresso de cada thread, alinhado para evitar falso
 * compartilhamento entre threads */
typedef struct {
    double soma;
    long long amostras;
    char preenchimento[64 - sizeof(double) - sizeof(long long)];
} Progresso;

/* Mensagem de backup enviada ao vizinho */
typedef struct {
    double soma;      /* soma parcial ate o momento */
    double amostras;  /* amostras ja processadas */
    double final;     /* 1.0 na ultima mensagem */
} Backup;

typedef enum { MODO_CPU, MODO_CPU_SEM_BACKUP, MODO_GPU } Modo;

static const char *nome_modo(Modo m)
{
    switch (m) {
    case MODO_CPU: return "cpu";
    case MODO_CPU_SEM_BACKUP: return "cpu_sem_backup";
    case MODO_GPU: return "gpu";
    }
    return "?";
}

/* --------------------------------------------------
 * Deteccao da GPU: existe dispositivo E o codigo realmente
 * executa nele (nao no host, como fallback)?
 * -------------------------------------------------- */
static int gpu_disponivel(void)
{
    if (omp_get_num_devices() <= 0) return 0;
    int no_host = 1;
#pragma omp target map(from : no_host)
    {
        no_host = omp_is_initial_device();
    }
    return !no_host;
}

/* Soma dos trapezios [ini, fim) na CPU */
static double soma_trapezios(size_t ini, size_t fim, double h)
{
    double s = 0.0;
    for (size_t i = ini; i < fim; i++) {
        double x0 = A_INTERVALO + i * h;
        s += 0.5 * h * (f(x0) + f(x0 + h));
    }
    return s;
}

/* Soma dos trapezios [ini, fim) na GPU */
static double soma_trapezios_gpu(size_t ini, size_t fim, double h)
{
    double s = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : s) \
    map(tofrom : s)
    for (size_t i = ini; i < fim; i++) {
        double x0 = A_INTERVALO + i * h;
        s += 0.5 * h * (f(x0) + f(x0 + h));
    }
    return s;
}

typedef struct {
    double soma_local;
    double tempo_calculo;
    long long mensagens_enviadas;
    long long mensagens_recebidas;
    double backup_soma;     /* ultimo valor recebido do vizinho */
    double backup_amostras;
} ResultadoLocal;

/* --------------------------------------------------
 * Calcula a parcela [ini, fim) do processo
 * -------------------------------------------------- */
static ResultadoLocal calcula_parcela(size_t ini, size_t fim, double h,
                                      Modo modo, int rank, int processos,
                                      int threads)
{
    ResultadoLocal res;
    memset(&res, 0, sizeof(res));

    size_t total_local = fim - ini;
    size_t total_blocos =
        (total_local + AMOSTRAS_CHECKPOINT - 1) / AMOSTRAS_CHECKPOINT;

    Progresso *progresso = calloc(threads, sizeof(Progresso));
    size_t proxima_pegada = 0; /* contador compartilhado de trabalho */
    int calculadoras_ativas = threads - 1;

    int direita = (rank + 1) % processos;            /* para quem envio */
    int esquerda = (rank - 1 + processos) % processos; /* de quem recebo */

    double t0 = MPI_Wtime();

#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();

        if (tid != 0) {
            /* ==================== THREADS DE CALCULO ==================== */
            double minha_soma = 0.0;
            long long minhas_amostras = 0;

            if (modo == MODO_GPU) {
                /* Uma thread envia lotes para a GPU; ao fim de cada lote
                 * publica o progresso, que a thread 0 repassa ao vizinho. */
                if (tid == 1) {
                    size_t tam_lote = (total_local + LOTES_GPU - 1) / LOTES_GPU;
                    tam_lote = ((tam_lote + AMOSTRAS_CHECKPOINT - 1) /
                                AMOSTRAS_CHECKPOINT) * AMOSTRAS_CHECKPOINT;
                    if (tam_lote == 0) tam_lote = AMOSTRAS_CHECKPOINT;

                    for (size_t a = ini; a < fim; a += tam_lote) {
                        size_t b = a + tam_lote < fim ? a + tam_lote : fim;
                        minha_soma += soma_trapezios_gpu(a, b, h);
                        minhas_amostras += (long long)(b - a);
#pragma omp atomic write
                        progresso[tid].soma = minha_soma;
#pragma omp atomic write
                        progresso[tid].amostras = minhas_amostras;
                    }
                }
            } else {
                /* CPU: distribuicao dinamica de "pegadas" de trabalho;
                 * o progresso e publicado a cada bloco de 100 amostras. */
                while (1) {
                    size_t pegada;
#pragma omp atomic capture
                    pegada = proxima_pegada++;

                    size_t bloco_ini = pegada * BLOCOS_POR_PEGADA;
                    if (bloco_ini >= total_blocos) break;
                    size_t bloco_fim = bloco_ini + BLOCOS_POR_PEGADA;
                    if (bloco_fim > total_blocos) bloco_fim = total_blocos;

                    for (size_t bl = bloco_ini; bl < bloco_fim; bl++) {
                        size_t a = ini + bl * AMOSTRAS_CHECKPOINT;
                        size_t b = a + AMOSTRAS_CHECKPOINT;
                        if (b > fim) b = fim;

                        minha_soma += soma_trapezios(a, b, h);
                        minhas_amostras += (long long)(b - a);
#pragma omp atomic write
                        progresso[tid].soma = minha_soma;
#pragma omp atomic write
                        progresso[tid].amostras = minhas_amostras;
                    }
                }
            }

#pragma omp atomic update seq_cst
            calculadoras_ativas--;
        } else {
            /* ================= THREAD 0: COMUNICACAO ================= */
            Backup envio, recebido;
            MPI_Request req_envio = MPI_REQUEST_NULL;
            long long ultimo_enviado = 0;
            int recebeu_final = (modo == MODO_CPU_SEM_BACKUP);
            int enviou_final = (modo == MODO_CPU_SEM_BACKUP);

            while (!enviou_final || !recebeu_final) {
                int ativas;
#pragma omp atomic read seq_cst
                ativas = calculadoras_ativas;

                /* Le o progresso de todas as threads de calculo */
                double soma = 0.0;
                long long amostras = 0;
                for (int t = 1; t < threads; t++) {
                    double s;
                    long long a;
#pragma omp atomic read
                    s = progresso[t].soma;
#pragma omp atomic read
                    a = progresso[t].amostras;
                    soma += s;
                    amostras += a;
                }

                if (modo != MODO_CPU_SEM_BACKUP) {
                    /* Envio: so quando o envio anterior ja terminou */
                    int livre = 1;
                    if (req_envio != MPI_REQUEST_NULL)
                        MPI_Test(&req_envio, &livre, MPI_STATUS_IGNORE);

                    if (livre && !enviou_final) {
                        int terminou = ativas == 0;
                        if (terminou ||
                            amostras - ultimo_enviado >= AMOSTRAS_CHECKPOINT) {
                            if (terminou) {
                                /* releitura apos o fim: valor definitivo */
                                soma = 0.0;
                                amostras = 0;
                                for (int t = 1; t < threads; t++) {
                                    double s;
                                    long long a;
#pragma omp atomic read seq_cst
                                    s = progresso[t].soma;
#pragma omp atomic read seq_cst
                                    a = progresso[t].amostras;
                                    soma += s;
                                    amostras += a;
                                }
                            }
                            envio.soma = soma;
                            envio.amostras = (double)amostras;
                            envio.final = terminou ? 1.0 : 0.0;
                            MPI_Isend(&envio, 3, MPI_DOUBLE, direita,
                                      TAG_BACKUP, MPI_COMM_WORLD, &req_envio);
                            ultimo_enviado = amostras;
                            res.mensagens_enviadas++;
                            if (terminou) enviou_final = 1;
                        }
                    }

                    /* Recebimento: guarda o ultimo valor do vizinho */
                    int chegou = 1;
                    while (chegou && !recebeu_final) {
                        MPI_Iprobe(esquerda, TAG_BACKUP, MPI_COMM_WORLD,
                                   &chegou, MPI_STATUS_IGNORE);
                        if (chegou) {
                            MPI_Recv(&recebido, 3, MPI_DOUBLE, esquerda,
                                     TAG_BACKUP, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
                            res.backup_soma = recebido.soma;
                            res.backup_amostras = recebido.amostras;
                            res.mensagens_recebidas++;
                            if (recebido.final == 1.0) recebeu_final = 1;
                        }
                    }
                } else if (ativas == 0) {
                    break;
                }

                /* Pausa curta: evita que a thread de comunicacao dispute
                 * o nucleo e as linhas de cache com as threads de calculo */
                if (!enviou_final || !recebeu_final || ativas > 0) {
                    struct timespec pausa = {0, PAUSA_COMUNICACAO_NS};
                    nanosleep(&pausa, NULL);
                }
            }

            if (req_envio != MPI_REQUEST_NULL)
                MPI_Wait(&req_envio, MPI_STATUS_IGNORE);
        }
    }

    res.tempo_calculo = MPI_Wtime() - t0;

    for (int t = 1; t < threads; t++) res.soma_local += progresso[t].soma;

    free(progresso);
    return res;
}

int main(int argc, char *argv[])
{
    int fornecido;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &fornecido);

    int rank, processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processos);

    if (fornecido < MPI_THREAD_FUNNELED) {
        if (rank == RAIZ)
            fprintf(stderr, "A biblioteca MPI nao suporta "
                            "MPI_THREAD_FUNNELED.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Rotulo opcional da execucao (ex.: "cpu", "gpu"), usado no nome do JSON */
    const char *rotulo = argc > 1 ? argv[1] : "";

    /* Pelo menos 2 threads: 1 de comunicacao + 1 de calculo */
    int threads = omp_get_max_threads();
    if (threads < 2) threads = 2;

    /* Deteccao automatica da GPU em cada processo */
    MPI_Comm comm_no;
    int rank_no;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &comm_no);
    MPI_Comm_rank(comm_no, &rank_no);
    MPI_Comm_free(&comm_no);
    if (omp_get_num_devices() > 0)
        omp_set_default_device(rank_no % omp_get_num_devices());

    int tem_gpu = gpu_disponivel();
    int processos_com_gpu = 0;
    MPI_Allreduce(&tem_gpu, &processos_com_gpu, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    char no[MPI_MAX_PROCESSOR_NAME];
    int tam_nome;
    memset(no, 0, sizeof(no));
    MPI_Get_processor_name(no, &tam_nome);

    char *nomes = NULL;
    int *gpus = NULL;
    if (rank == RAIZ) {
        nomes = malloc((size_t)processos * MPI_MAX_PROCESSOR_NAME);
        gpus = malloc(processos * sizeof(int));
    }
    MPI_Gather(no, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, nomes,
               MPI_MAX_PROCESSOR_NAME, MPI_CHAR, RAIZ, MPI_COMM_WORLD);
    MPI_Gather(&tem_gpu, 1, MPI_INT, gpus, 1, MPI_INT, RAIZ, MPI_COMM_WORLD);

    FILE *json = NULL;
    if (rank == RAIZ) {
        char nome_arquivo[128];
        if (rotulo[0] != '\0')
            snprintf(nome_arquivo, sizeof(nome_arquivo),
                     "resultados_%s_np%d.json", rotulo, processos);
        else
            snprintf(nome_arquivo, sizeof(nome_arquivo),
                     "resultados_np%d.json", processos);
        json = fopen(nome_arquivo, "w");
        if (json == NULL) {
            fprintf(stderr, "Erro ao abrir %s\n", nome_arquivo);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Integral por trapezios: %d processos x %d threads "
               "(1 comunicacao + %d calculo)\n",
               processos, threads, threads - 1);
        printf("Processos com GPU: %d de %d\n", processos_com_gpu, processos);

        fprintf(json, "{\n  \"rotulo\": \"%s\",\n", rotulo);
        fprintf(json, "  \"processos\": %d,\n  \"threads\": %d,\n",
                processos, threads);
        fprintf(json, "  \"processos_com_gpu\": %d,\n", processos_com_gpu);
        fprintf(json, "  \"mpi_thread\": \"%s\",\n",
                fornecido == MPI_THREAD_MULTIPLE     ? "MULTIPLE"
                : fornecido == MPI_THREAD_SERIALIZED ? "SERIALIZED"
                                                     : "FUNNELED");
        fprintf(json, "  \"nos\": [");
        for (int p = 0; p < processos; p++)
            fprintf(json, "{\"rank\": %d, \"no\": \"%s\", \"gpu\": %s}%s", p,
                    nomes + p * MPI_MAX_PROCESSOR_NAME,
                    gpus[p] ? "true" : "false",
                    p + 1 < processos ? ", " : "");
        fprintf(json, "],\n");
        fprintf(json, "  \"valor_exato\": %.17g,\n", VALOR_EXATO);
        fprintf(json, "  \"resultados\": [\n");

        printf("%12s %-15s %12s %12s %12s %10s %20s %10s %8s\n", "N", "modo",
               "total (s)", "calculo (s)", "reduce (s)", "msgs", "integral",
               "erro rel", "backup");
    }

    /* GPU so e usada se TODOS os processos tiverem uma */
    Modo modos[3] = {MODO_CPU, MODO_CPU_SEM_BACKUP, MODO_GPU};
    int total_modos = processos_com_gpu == processos ? 3 : 2;

    int primeiro = 1;

    for (size_t q = 0; q < total_quantidades; q++) {
        size_t n = quantidades[q];
        double h = (B_INTERVALO - A_INTERVALO) / (double)n;

        /* Divisao em blocos contiguos (o resto vai para os primeiros) */
        size_t base = n / processos, resto = n % processos;
        size_t ini = rank * base + ((size_t)rank < resto ? (size_t)rank : resto);
        size_t fim = ini + base + ((size_t)rank < resto ? 1 : 0);

        for (int m = 0; m < total_modos; m++) {
            Modo modo = modos[m];

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            ResultadoLocal r = calcula_parcela(ini, fim, h, modo, rank,
                                               processos, threads);

            double t1 = MPI_Wtime();
            double integral = 0.0;
            MPI_Reduce(&r.soma_local, &integral, 1, MPI_DOUBLE, MPI_SUM, RAIZ,
                       MPI_COMM_WORLD);
            double t2 = MPI_Wtime();

            /* O backup guardado em cada processo deve coincidir com a
             * soma final do vizinho da esquerda */
            double soma_esquerda = 0.0;
            int esquerda = (rank - 1 + processos) % processos;
            int direita = (rank + 1) % processos;
            MPI_Sendrecv(&r.soma_local, 1, MPI_DOUBLE, direita, 5,
                         &soma_esquerda, 1, MPI_DOUBLE, esquerda, 5,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int backup_ok = modo == MODO_CPU_SEM_BACKUP ||
                            fabs(r.backup_soma - soma_esquerda) <= 1e-12;
            int todos_ok = 0;
            MPI_Reduce(&backup_ok, &todos_ok, 1, MPI_INT, MPI_LAND, RAIZ,
                       MPI_COMM_WORLD);

            double tempos[2] = {t1 - t0, t2 - t1}, tempos_max[2];
            MPI_Reduce(tempos, tempos_max, 2, MPI_DOUBLE, MPI_MAX, RAIZ,
                       MPI_COMM_WORLD);
            long long msgs = r.mensagens_enviadas, msgs_total = 0;
            MPI_Reduce(&msgs, &msgs_total, 1, MPI_LONG_LONG, MPI_SUM, RAIZ,
                       MPI_COMM_WORLD);

            if (rank == RAIZ) {
                double total = tempos_max[0] + tempos_max[1];
                double erro_abs = fabs(integral - VALOR_EXATO);
                double erro_rel = erro_abs / VALOR_EXATO;

                printf("%12zu %-15s %12.6f %12.6f %12.6f %10lld %20.15f "
                       "%10.2e %8s\n",
                       n, nome_modo(modo), total, tempos_max[0],
                       tempos_max[1], msgs_total, integral, erro_rel,
                       modo == MODO_CPU_SEM_BACKUP ? "-"
                       : todos_ok                  ? "ok"
                                                   : "ERRO");

                fprintf(json,
                        "%s    {\"n\": %zu, \"modo\": \"%s\", "
                        "\"tempo_total_s\": %.9f, \"tempo_calculo_s\": %.9f, "
                        "\"tempo_reduce_s\": %.9f, \"mensagens_backup\": %lld, "
                        "\"integral\": %.17g, \"erro_abs\": %.6e, "
                        "\"erro_rel\": %.6e, \"backup_ok\": %s}",
                        primeiro ? "" : ",\n", n, nome_modo(modo), total,
                        tempos_max[0], tempos_max[1], msgs_total, integral,
                        erro_abs, erro_rel, todos_ok ? "true" : "false");
                primeiro = 0;
            }
        }
    }

    if (rank == RAIZ) {
        fprintf(json, "\n  ]\n}\n");
        fclose(json);
        printf("Resultados salvos (rotulo \"%s\", %d processos)\n", rotulo,
               processos);
        free(nomes);
        free(gpus);
    }

    MPI_Finalize();
    return 0;
}
