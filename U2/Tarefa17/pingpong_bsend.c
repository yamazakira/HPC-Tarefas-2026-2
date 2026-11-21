/*
 * Tarefa 17 - Ping-pong com MPI_Bsend (modo BUFFERIZADO)
 *
 * MPI_Bsend copia a mensagem para um buffer fornecido pelo USUARIO
 * (registrado com MPI_Buffer_attach) e retorna imediatamente, sem
 * depender do receptor. O envio real acontece depois, a partir desse
 * buffer. Vantagem: o emissor nunca bloqueia esperando o receptor.
 * Custo: uma copia extra de memoria (visivel em mensagens grandes) e a
 * necessidade de dimensionar o buffer (tamanho + MPI_BSEND_OVERHEAD
 * por mensagem pendente); se faltar espaco, o MPI_Bsend falha.
 *
 * compilar: mpicc -O2 pingpong_bsend.c -o pingpong_bsend
 * executar: mpirun -np 2 ./pingpong_bsend
 */
#include "pingpong_comum.h"

static void *buffer_bsend = NULL;

static void prepara_modo(size_t maior_mensagem)
{
    /* Espaco para 2 mensagens pendentes do maior tamanho (folga) */
    int tamanho = (int)(2 * (maior_mensagem + MPI_BSEND_OVERHEAD));
    buffer_bsend = malloc(tamanho);
    if (buffer_bsend == NULL) {
        fprintf(stderr, "Erro ao alocar buffer do MPI_Bsend.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Buffer_attach(buffer_bsend, tamanho);
}

static void encerra_modo(void)
{
    void *endereco;
    int tamanho;
    /* Bloqueia ate que todas as mensagens bufferizadas sejam enviadas */
    MPI_Buffer_detach(&endereco, &tamanho);
    free(buffer_bsend);
}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank)
{
    (void)recepcao2;

    for (int i = 0; i < repeticoes; i++) {
        if (rank == 0) {
            MPI_Bsend(envio, bytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
            MPI_Recv(recepcao, bytes, MPI_CHAR, 1, TAG_PONG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recepcao, bytes, MPI_CHAR, 0, TAG_PING, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Bsend(recepcao, bytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[])
{
    return executa_pingpong(argc, argv, "bsend", "MPI_Bsend");
}
