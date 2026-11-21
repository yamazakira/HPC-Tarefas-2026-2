/*
 * Tarefa 17 - Ping-pong com MPI_Send (modo PADRAO)
 *
 * MPI_Send e o envio "padrao": a biblioteca decide se copia a mensagem
 * para um buffer interno (protocolo eager, comum em mensagens pequenas)
 * ou se espera o receptor postar o MPI_Recv correspondente (protocolo
 * rendezvous, comum em mensagens grandes). Ele retorna quando o buffer
 * de envio pode ser reutilizado, o que pode ou nao significar que a
 * mensagem ja foi recebida.
 *
 * compilar: mpicc -O2 pingpong_send.c -o pingpong_send
 * executar: mpirun -np 2 ./pingpong_send
 */
#include "pingpong_comum.h"

static void prepara_modo(size_t maior_mensagem) { (void)maior_mensagem; }
static void encerra_modo(void) {}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank)
{
    (void)recepcao2;

    for (int i = 0; i < repeticoes; i++) {
        if (rank == 0) {
            /* ping: 0 -> 1 */
            MPI_Send(envio, bytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
            /* pong: 1 -> 0 */
            MPI_Recv(recepcao, bytes, MPI_CHAR, 1, TAG_PONG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recepcao, bytes, MPI_CHAR, 0, TAG_PING, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            /* responde imediatamente com a mesma mensagem */
            MPI_Send(recepcao, bytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[])
{
    return executa_pingpong(argc, argv, "send", "MPI_Send");
}
