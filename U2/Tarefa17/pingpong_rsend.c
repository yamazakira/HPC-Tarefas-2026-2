#include "pingpong_comum.h"

static void prepara_modo(size_t maior_mensagem) { (void)maior_mensagem; }
static void encerra_modo(void) {}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank)
{
    MPI_Request req;

    if (rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD);

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
