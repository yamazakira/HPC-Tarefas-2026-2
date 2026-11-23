#include "pingpong_comum.h"

static void prepara_modo(size_t maior_mensagem) { (void)maior_mensagem; }
static void encerra_modo(void) {}

static void troca_mensagens(char *envio, char *recepcao, char *recepcao2,
                            int bytes, int repeticoes, int rank)
{
    (void)recepcao2;

    for (int i = 0; i < repeticoes; i++) {
        if (rank == 0) {
            MPI_Ssend(envio, bytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
            MPI_Recv(recepcao, bytes, MPI_CHAR, 1, TAG_PONG, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recepcao, bytes, MPI_CHAR, 0, TAG_PING, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Ssend(recepcao, bytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char *argv[])
{
    return executa_pingpong(argc, argv, "ssend", "MPI_Ssend");
}
