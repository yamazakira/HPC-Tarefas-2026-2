#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

bool isPrimo(int numero) {
  if (numero < 2) {
    return false;
  }

  for (int i = 2; i * i <= numero; i++) {
    if (numero % i == 0) {
      return false;
    }
  }

  return true;
}

double contarSequencial(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[SEQ] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloPrivado(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for private(primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[PRIVATE] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloFirstPrivado(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for firstprivate(primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[FIRSTPRIVATE] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloLastPrivado(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for lastprivate(primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[LASTPRIVATE] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloFirstLastPrivado(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for firstprivate(primos_encontrados) lastprivate(primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[FIRST+LASTPRIVATE] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloDefaultNone(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for default(none) shared(num_inicio, num_fim) private(primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[DEFAULT(NONE)] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

double contarParaleloReduction(int num_inicio, int num_fim) {
  int numeros_verificados = num_fim - num_inicio;
  int primos_encontrados = 0;

  double tempo_i = omp_get_wtime();

#pragma omp parallel for reduction(+ : primos_encontrados)
  for (int i = num_inicio + 1; i < num_fim; i++) {
    if (isPrimo(i)) {
      primos_encontrados++;
    }
  }

  double tempo_f = omp_get_wtime();
  double tempo_gasto = tempo_f - tempo_i;

  printf("[REDUCTION] | # Verificados: %d | Primos: %d | Tempo: %f\n",
         numeros_verificados, primos_encontrados, tempo_gasto);

  return tempo_gasto;
}

int main() {
  int inicio = 2;

  const size_t iteracoes[] = {100, 1000, 10000, 100000, 1000000, 10000000};

  const size_t quantidade = sizeof(iteracoes) / sizeof(iteracoes[0]);

  FILE *arquivo_saida = fopen("resultados7.csv", "w");

  if (arquivo_saida == NULL) {
    printf("Erro ao abrir o arquivo CSV.\n");
    return 1;
  }

  fprintf(arquivo_saida,
          "numeros_verificados,tempo_sequencial,tempo_privado,"
          "tempo_firstprivado,tempo_lastprivado,"
          "tempo_first_last_privado,tempo_default_none,tempo_reduction\n");

  for (size_t i = 0; i < quantidade; i++) {
    int fim = iteracoes[i];

    printf("\n====== BUSCANDO PRIMOS ATÉ %d ======\n", fim);

    double tempo_sequencial = contarSequencial(inicio, fim);
    double tempo_privado = contarParaleloPrivado(inicio, fim);
    double tempo_firstprivado = contarParaleloFirstPrivado(inicio, fim);
    double tempo_lastprivado = contarParaleloLastPrivado(inicio, fim);
    double tempo_first_last_privado =
        contarParaleloFirstLastPrivado(inicio, fim);
    double tempo_default_none = contarParaleloDefaultNone(inicio, fim);
    double tempo_reduction = contarParaleloReduction(inicio, fim);

    fprintf(arquivo_saida,
            "%d,%f,%f,%f,%f,%f,%f,%f\n",
            fim - inicio,
            tempo_sequencial,
            tempo_privado,
            tempo_firstprivado,
            tempo_lastprivado,
            tempo_first_last_privado,
            tempo_default_none,
            tempo_reduction);
  }

  fclose(arquivo_saida);

  printf("\nResultados salvos em resultados7.csv\n");

  return 0;
}