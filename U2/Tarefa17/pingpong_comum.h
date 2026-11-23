#ifndef PINGPONG_COMUM_H
#define PINGPONG_COMUM_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_PING 10
#define TAG_PONG 20

#define AQUECIMENTO 10

static const size_t tamanhos[] = {
    8,       16,      32,      64,      128,     256,     512,
    1024,    2048,    4096,    8192,    16384,   32768,   65536,
    131072,  262144,  524288,  1048576, 2097152, 4194304, 8388608,
    16777216,
};
static const size_t quantidade_tamanhos =
    sizeof(tamanhos) / sizeof(tamanhos[0]);

static size_t maior_tamanho(void)
{
    size_t m = 0;
    for (size_t i = 0; i < quantidade_tamanhos; i++)
        if (tamanhos[i] > m) m = tamanhos[i];
    return m;
}

static int repeticoes_para(size_t bytes)
{
    if (bytes <= 65536) return 1000;
    size_t r = (64u * 1024u * 1024u) / bytes;
    return r < 20 ? 20 : (int)r;
}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank);

static void prepara_modo(size_t maior_mensagem);
static void encerra_modo(void);

static int executa_pingpong(int argc, char *argv[], const char *modo,
                            const char *funcao)
{
    MPI_Init(&argc, &argv);

    int rank, processos;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processos);

    if (processos != 2) {
        if (rank == 0)
            fprintf(stderr, "Este programa exige exatamente 2 processos "
                            "(recebeu %d).\n", processos);
        MPI_Finalize();
        return 1;
    }

    char no[MPI_MAX_PROCESSOR_NAME], no_outro[MPI_MAX_PROCESSOR_NAME];
    int tam_nome;
    memset(no, 0, sizeof(no));
    MPI_Get_processor_name(no, &tam_nome);
    if (rank == 0) {
        MPI_Recv(no_outro, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Send(no, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    size_t maior = maior_tamanho();
    char *envio = malloc(maior);
    char *recepcao = malloc(maior);
    char *recepcao2 = malloc(maior);
    if (!envio || !recepcao || !recepcao2) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (size_t i = 0; i < maior; i++) envio[i] = (char)(i % 251);

    prepara_modo(maior);

    char nome_arquivo[64];
    snprintf(nome_arquivo, sizeof(nome_arquivo), "resultados_%s.json", modo);

    FILE *json = NULL;
    if (rank == 0) {
        json = fopen(nome_arquivo, "w");
        if (json == NULL) {
            fprintf(stderr, "Erro ao abrir %s\n", nome_arquivo);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int mesmo_no = strcmp(no, no_outro) == 0;
        fprintf(json, "{\n");
        fprintf(json, "  \"modo\": \"%s\",\n", modo);
        fprintf(json, "  \"funcao\": \"%s\",\n", funcao);
        fprintf(json, "  \"no_processo_0\": \"%s\",\n", no);
        fprintf(json, "  \"no_processo_1\": \"%s\",\n", no_outro);
        fprintf(json, "  \"comunicacao\": \"%s\",\n",
                mesmo_no ? "intra-no" : "inter-no");
        fprintf(json, "  \"resultados\": [\n");

        printf("[%s] processo 0 em %s, processo 1 em %s (%s)\n", funcao, no,
               no_outro, mesmo_no ? "intra-no" : "inter-no");
        printf("%10s %8s %14s %14s %14s\n", "bytes", "reps", "total (s)",
               "latencia(us)", "banda (MB/s)");
    }

    for (size_t t = 0; t < quantidade_tamanhos; t++) {
        int bytes = (int)tamanhos[t];
        int repeticoes = repeticoes_para(tamanhos[t]);

        MPI_Barrier(MPI_COMM_WORLD);
        troca_mensagens(envio, recepcao, recepcao2, bytes, AQUECIMENTO, rank);

        memset(recepcao, 0, bytes);
        MPI_Barrier(MPI_COMM_WORLD);

        double inicio = MPI_Wtime();
        troca_mensagens(envio, recepcao, recepcao2, bytes, repeticoes, rank);
        double fim = MPI_Wtime();

        if (rank == 0) {

            int ok = memcmp(envio, recepcao, bytes) == 0;

            double total = fim - inicio;
            double ida_volta = total / repeticoes;
            double latencia = ida_volta / 2.0;
            double banda = bytes / latencia / 1e6;

            printf("%10d %8d %14.6f %14.3f %14.2f %s\n", bytes, repeticoes,
                   total, latencia * 1e6, banda, ok ? "" : "ERRO");

            fprintf(json,
                    "    {\"bytes\": %d, \"repeticoes\": %d, "
                    "\"tempo_total_s\": %.9f, \"tempo_ida_volta_us\": %.4f, "
                    "\"latencia_us\": %.4f, \"banda_mbs\": %.4f, "
                    "\"valido\": %s}%s\n",
                    bytes, repeticoes, total, ida_volta * 1e6, latencia * 1e6,
                    banda, ok ? "true" : "false",
                    t + 1 < quantidade_tamanhos ? "," : "");
        }
    }

    if (rank == 0) {
        fprintf(json, "  ]\n}\n");
        fclose(json);
        printf("Resultados salvos em %s\n", nome_arquivo);
    }

    encerra_modo();

    free(envio);
    free(recepcao);
    free(recepcao2);

    MPI_Finalize();
    return 0;
}

#endif
