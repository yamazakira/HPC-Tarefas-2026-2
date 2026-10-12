#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_NOME 100
#define NUM_ARQUIVOS 5
#define NUM_ALGARISMOS 10

typedef struct No {

    int arquivo[MAX_NOME];
    int tamanho_arquivo;

    struct No *proximo;

} No;

int histograma[NUM_ALGARISMOS] = {0};

omp_lock_t lock_histograma;


// Cria um novo nó e preenche o arquivo com números aleatórios
No* criar_no()
{
    No *novo = (No *) malloc(sizeof(No));

    if (novo == NULL) {
        printf("Erro na alocacao de memoria.\n");
        exit(1);
    }

    // Define aleatoriamente o tamanho do arquivo
    novo->tamanho_arquivo = 1 + rand() % MAX_NOME;

    // Preenche o arquivo com algarismos de 0 a 9
    for (int i = 0; i < novo->tamanho_arquivo; i++) {
        novo->arquivo[i] = rand() % 10;
    }

    novo->proximo = NULL;

    return novo;
}


// Insere um novo arquivo no final da lista
void inserir(No **lista)
{
    No *novo = criar_no();

    if (*lista == NULL) {
        *lista = novo;
        return;
    }

    No *atual = *lista;

    while (atual->proximo != NULL) {
        atual = atual->proximo;
    }

    atual->proximo = novo;
}


// Imprime os arquivos da lista
void imprimir_lista(No *lista)
{
    No *atual = lista;

    int numero_arquivo = 1;

    while (atual != NULL) {

        printf("Arquivo %d (tamanho = %d):\n",
               numero_arquivo,
               atual->tamanho_arquivo);

        for (int i = 0; i < atual->tamanho_arquivo; i++) {
            printf("%d ", atual->arquivo[i]);
        }

        printf("\n\n");

        atual = atual->proximo;
        numero_arquivo++;
    }
}

void processar_arquivo(No *no)
{
    int local[NUM_ALGARISMOS] = {0};

    for (int i = 0; i < no->tamanho_arquivo; i++) {
        int algarismo = no->arquivo[i];
        local[algarismo]++;
    }

    omp_set_lock(&lock_histograma);

    for (int d = 0; d < NUM_ALGARISMOS; d++) {
        histograma[d] += local[d];
    }

    omp_unset_lock(&lock_histograma);
}

void processar_lista_paralelo(No *lista)
{
    omp_init_lock(&lock_histograma);

    #pragma omp parallel
    {
        #pragma omp single
        {
            No *atual = lista;

            while (atual != NULL) {

                No *no_atual = atual;

                #pragma omp task firstprivate(no_atual)
                {
                    processar_arquivo(no_atual);
                }

                atual = atual->proximo;
            }

            #pragma omp taskwait
        }
    }

    omp_destroy_lock(&lock_histograma);
}

void imprimir_histograma()
{
    printf("Histograma global de ocorrencias:\n");

    for (int d = 0; d < NUM_ALGARISMOS; d++) {
        printf("Algarismo %d: %d ocorrencia(s)\n", d, histograma[d]);
    }
}


// Libera a memória da lista
void liberar_lista(No *lista)
{
    No *atual = lista;

    while (atual != NULL) {

        No *temp = atual;
        atual = atual->proximo;

        free(temp);
    }
}


int main()
{
    No *lista = NULL;

    srand(time(NULL));

    // Cria NUM_ARQUIVOS arquivos
    for (int i = 0; i < NUM_ARQUIVOS; i++) {
        inserir(&lista);
    }

    imprimir_lista(lista);

    processar_lista_paralelo(lista);

    imprimir_histograma();

    liberar_lista(lista);

    return 0;
}