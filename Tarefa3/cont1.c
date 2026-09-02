#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <semaphore.h>

#define NUM_PROCESSOS 4
#define INCREMENTOS 1000000

int main()
{
    int *contador = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (contador == MAP_FAILED)
    {
        perror("Erro ao criar memória compartilhada");
        exit(EXIT_FAILURE);
    }
    *contador = 0;

    // semáforo compartilhado entre processos
    sem_t *sem = mmap(NULL, sizeof(sem_t), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sem == MAP_FAILED)
    {
        perror("Erro ao criar semáforo");
        exit(EXIT_FAILURE);
    }
    sem_init(sem, 1, 1);

    printf("Contador inicial: %d\n", *contador);

    for (int i = 0; i < NUM_PROCESSOS; i++)
    {
        pid_t pid = fork();
        if (pid < 0)
        {
            perror("Erro ao criar processo");
            exit(EXIT_FAILURE);
        }
        if (pid == 0)
        {
            // Código executado pelo processo filho
            for (int j = 0; j < INCREMENTOS; j++)
            {
                sem_wait(sem); // trava
                (*contador)++;
                sem_post(sem); // destrava
            }
            printf("Filho %d terminou\n", getpid());
            exit(EXIT_SUCCESS);
        }
    }

    // Processo pai espera todos os filhos
    for (int i = 0; i < NUM_PROCESSOS; i++)
    {
        wait(NULL);
    }

    printf("\nContador no processo pai: %d\n", *contador);

    // Limpeza
    sem_destroy(sem);
    munmap(contador, sizeof(int));
    munmap(sem, sizeof(sem_t));

    return 0;
}