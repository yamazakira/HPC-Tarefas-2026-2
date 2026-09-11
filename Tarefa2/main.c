#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
  double inicializacao;
  double soma_dependente;
  double soma_independente;
} Resultado;

double obter_tempo_atual(void) {
  struct timespec tempo;

  clock_gettime(CLOCK_MONOTONIC, &tempo);

  return (double)tempo.tv_sec + (double)tempo.tv_nsec / 1e9;
}

double *criar_vetor(size_t tamanho) {
  double *vetor = malloc(tamanho * sizeof(double));

  if (vetor == NULL) {
    fprintf(stderr, "Erro ao alocar vetor\n");
    exit(EXIT_FAILURE);
  }

  return vetor;
}

void liberar_vetor(double *vetor) { free(vetor); }

void inicializar_vetor(double *vetor, size_t tamanho) {
  for (size_t i = 0; i < tamanho; i++) {
    vetor[i] = (double)i * 1.5 + 10.0;
  }
}

double somar_dependente(const double *vetor, size_t tamanho) {
  double soma = 0.0;

  for (size_t i = 0; i < tamanho; i++) {
    soma += vetor[i];
  }

  return soma;
}

double somar_independente(const double *vetor, size_t tamanho) {
  double soma1 = 0.0;
  double soma2 = 0.0;
  double soma3 = 0.0;
  double soma4 = 0.0;

  size_t i = 0;

  for (; i + 3 < tamanho; i += 4) {
    soma1 += vetor[i];
    soma2 += vetor[i + 1];
    soma3 += vetor[i + 2];
    soma4 += vetor[i + 3];
  }

  for (; i < tamanho; i++) {
    soma1 += vetor[i];
  }

  return soma1 + soma2 + soma3 + soma4;
}

double medir_inicializacao(double *vetor, size_t tamanho) {
  double inicio = obter_tempo_atual();

  inicializar_vetor(vetor, tamanho);

  double fim = obter_tempo_atual();

  return fim - inicio;
}

double medir_soma_dependente(const double *vetor, size_t tamanho,
                             double *resultado) {
  double inicio = obter_tempo_atual();

  *resultado = somar_dependente(vetor, tamanho);

  double fim = obter_tempo_atual();

  return fim - inicio;
}

double medir_soma_independente(const double *vetor, size_t tamanho,
                               double *resultado) {
  double inicio = obter_tempo_atual();

  *resultado = somar_independente(vetor, tamanho);

  double fim = obter_tempo_atual();

  return fim - inicio;
}

Resultado executar_testes(size_t tamanho) {
  Resultado resultado;
  double soma_dependente;
  double soma_independente;

  double *vetor = criar_vetor(tamanho);

  resultado.inicializacao = medir_inicializacao(vetor, tamanho);

  resultado.soma_dependente =
      medir_soma_dependente(vetor, tamanho, &soma_dependente);

  resultado.soma_independente =
      medir_soma_independente(vetor, tamanho, &soma_independente);

  printf("%zu,%.9f,%.9f,%.9f,%.2f,%.2f\n", tamanho, resultado.inicializacao,
         resultado.soma_dependente, resultado.soma_independente,
         soma_dependente, soma_independente);

  liberar_vetor(vetor);

  return resultado;
}

void executar_tamanhos(const size_t *tamanhos, size_t quantidade) {
  for (size_t i = 0; i < quantidade; i++) {
    executar_testes(tamanhos[i]);
  }
}

int main(void) {
  const size_t tamanhos[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};

  size_t quantidade = sizeof(tamanhos) / sizeof(tamanhos[0]);

  printf("tamanho,inicializacao,soma_dependente,"
         "soma_independente,resultado_dependente,"
         "resultado_independente\n");

  executar_tamanhos(tamanhos, quantidade);

  return 0;
}