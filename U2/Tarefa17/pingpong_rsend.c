/*
 * Tarefa 17 - Ping-pong com MPI_Rsend (modo READY / PRONTO)
 *
 * MPI_Rsend so e CORRETO se o receptor ja tiver postado o recebimento
 * correspondente antes do envio comecar. Com essa garantia, a
 * biblioteca pode pular o handshake e enviar os dados direto. Se o
 * recebimento nao estiver postado, o comportamento e indefinido (erro
 * ou perda da mensagem). Por isso o protocolo abaixo pre-posta os
 * recebimentos com MPI_Irecv:
 *
 *   processo 0: Irecv(pong i)  -> Rsend(ping i) -> Wait(pong i)
 *   processo 1: Wait(ping i)   -> Irecv(ping i+1) -> Rsend(pong i)
 *
 * - O processo 1 so faz Rsend(pong i) depois de receber o ping i, que
 *   o processo 0 so enviou apos postar Irecv(pong i).
 * - O processo 0 so faz Rsend(ping i+1) depois de receber o pong i, que
 *   o processo 1 so enviou apos postar Irecv(ping i+1).
 * - O primeiro Irecv do processo 1 e garantido por uma MPI_Barrier.
 * O processo 1 alterna entre dois buffers, pois recebe o ping i+1
 * enquanto ainda esta enviando o pong i.
 *
 * compilar: mpicc -O2 pingpong_rsend.c -o pingpong_rsend
 * executar: mpirun -np 2 ./pingpong_rsend
 */
#include "pingpong_comum.h"

static void prepara_modo(size_t maior_mensagem) { (void)maior_mensagem; }
static void encerra_modo(void) {}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank)
{
    MPI_Request req;

    if (rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD); /* processo 1 ja postou Irecv(ping 0) */

        for (int i = 0; i < repeticoes; i++) {
            MPI_Irecv(recepcao, bytes, MPI_CHAR, 1, TAG_PONG, MPI_COMM_WORLD,
                      &req);
            MPI_Rsend(envio, bytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
    } else {
        char *buffers[2] = {recepcao, recepcao2};
        int atual = 0;

        MPI_Irecv(buffers[atual], bytes, MPI_CHAR, 0, TAG_PING,
                  MPI_COMM_WORLD, &req);
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < repeticoes; i++) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            char *recebido = buffers[atual];

            /* pre-posta o proximo ping ANTES de responder */
            if (i + 1 < repeticoes) {
                atual = 1 - atual;
                MPI_Irecv(buffers[atual], bytes, MPI_CHAR, 0, TAG_PING,
                          MPI_COMM_WORLD, &req);
            }

            MPI_Rsend(recebido, bytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[])
{
    return executa_pingpong(argc, argv, "rsend", "MPI_Rsend");
}
