#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NOME 100
#define NUM_ARQUIVOS 5

typedef struct No {

    int arquivo[MAX_NOME];
    int tamanho_arquivo;

    struct No *proximo;

} No;


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

    liberar_lista(lista);

    return 0;
}