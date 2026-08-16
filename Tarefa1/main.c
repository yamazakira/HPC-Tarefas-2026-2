#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double ObterTempoAtual(void)
{
    return (double)clock() / CLOCKS_PER_SEC;
}

double *AlocarMatriz(size_t quantidade_linhas, size_t quantidade_colunas)
{
    double *matriz = malloc(quantidade_linhas * quantidade_colunas * sizeof(double));
    if (matriz == NULL)
    {
        fprintf(stderr, "Erro ao alocar matriz.\n");
        exit(EXIT_FAILURE);
    }
    return matriz;
}

double *AlocarVetor(size_t quantidade_elementos)
{
    double *vetor = malloc(quantidade_elementos * sizeof(double));
    if (vetor == NULL)
    {
        fprintf(stderr, "Erro ao alocar vetor.\n");
        exit(EXIT_FAILURE);
    }
    return vetor;
}

void InicializarMatriz(double *matriz, size_t quantidade_linhas, size_t quantidade_colunas)
{
    for (size_t indice_linha = 0; indice_linha < quantidade_linhas; ++indice_linha)
    {
        for (size_t indice_coluna = 0; indice_coluna < quantidade_colunas; ++indice_coluna)
        {
            matriz[indice_linha * quantidade_colunas + indice_coluna] =
                1.0 + (double)((indice_linha + indice_coluna) % 100) / 100.0;
        }
    }
}

void InicializarVetor(double *vetor, size_t quantidade_elementos)
{
    for (size_t indice = 0; indice < quantidade_elementos; ++indice)
    {
        vetor[indice] = 1.0 + (double)(indice % 100) / 100.0;
    }
}

void ZerarVetor(double *vetor, size_t quantidade_elementos)
{
    for (size_t indice = 0; indice < quantidade_elementos; ++indice)
    {
        vetor[indice] = 0.0;
    }
}

void MultiplicarPorLinhas(
    const double *matriz,
    const double *vetor_entrada,
    double *vetor_saida,
    size_t quantidade_linhas,
    size_t quantidade_colunas)
{
    for (size_t indice_linha = 0; indice_linha < quantidade_linhas; ++indice_linha)
    {
        double soma_linha = 0.0;

        for (size_t indice_coluna = 0; indice_coluna < quantidade_colunas; ++indice_coluna)
        {
            soma_linha += matriz[indice_linha * quantidade_colunas + indice_coluna] *
                          vetor_entrada[indice_coluna];
        }

        vetor_saida[indice_linha] = soma_linha;
    }
}

void MultiplicarPorColunas(
    const double *matriz,
    const double *vetor_entrada,
    double *vetor_saida,
    size_t quantidade_linhas,
    size_t quantidade_colunas)
{
    ZerarVetor(vetor_saida, quantidade_linhas);

    for (size_t indice_coluna = 0; indice_coluna < quantidade_colunas; ++indice_coluna)
    {
        const double valor_vetor_entrada = vetor_entrada[indice_coluna];

        for (size_t indice_linha = 0; indice_linha < quantidade_linhas; ++indice_linha)
        {
            vetor_saida[indice_linha] +=
                matriz[indice_linha * quantidade_colunas + indice_coluna] * valor_vetor_entrada;
        }
    }
}

// calcular a diferença entre as saídas pra ver se teve algum erro
// de cálculo entre as matrixes
double CalcularErroMaximo(
    const double *primeiro_vetor,
    const double *segundo_vetor,
    size_t quantidade_elementos)
{
    double maior_erro = 0.0;

    for (size_t indice = 0; indice < quantidade_elementos; ++indice)
    {
        double erro_atual = primeiro_vetor[indice] - segundo_vetor[indice];
        if (erro_atual < 0.0)
        {
            erro_atual = -erro_atual;
        }
        if (erro_atual > maior_erro)
        {
            maior_erro = erro_atual;
        }
    }

    return maior_erro;
}

double MedirLinhas(
    const double *matriz,
    const double *vetor_entrada,
    double *vetor_saida,
    size_t quantidade_linhas,
    size_t quantidade_colunas,
    size_t repeticoes)
{
    double tempo_inicio = ObterTempoAtual();

    for (size_t i = 0; i < repeticoes; ++i)
    {
        MultiplicarPorLinhas(
            matriz, vetor_entrada, vetor_saida, quantidade_linhas, quantidade_colunas);
    }

    double tempo_fim = ObterTempoAtual();
    return tempo_fim - tempo_inicio;
}

double MedirColunas(
    const double *matriz,
    const double *vetor_entrada,
    double *vetor_saida,
    size_t quantidade_linhas,
    size_t quantidade_colunas,
    size_t repeticoes)
{
    double tempo_inicio = ObterTempoAtual();

    for (size_t i = 0; i < repeticoes; ++i)
    {
        MultiplicarPorColunas(
            matriz, vetor_entrada, vetor_saida, quantidade_linhas, quantidade_colunas);
    }

    double tempo_fim = ObterTempoAtual();
    return tempo_fim - tempo_inicio;
}

/* Executa um teste para uma matriz quadrada n x n. */
void ExecutarTeste(size_t repeticoes, size_t tamanho_matriz, FILE *arquivo_saida)
{
    double *matriz = AlocarMatriz(tamanho_matriz, tamanho_matriz);
    double *vetor_entrada = AlocarVetor(tamanho_matriz);
    double *vetor_saida_linhas = AlocarVetor(tamanho_matriz);
    double *vetor_saida_colunas = AlocarVetor(tamanho_matriz);

    InicializarMatriz(matriz, tamanho_matriz, tamanho_matriz);
    InicializarVetor(vetor_entrada, tamanho_matriz);

    double tempo_linhas = MedirLinhas(
        matriz, vetor_entrada, vetor_saida_linhas, tamanho_matriz, tamanho_matriz, repeticoes);

    double tempo_colunas = MedirColunas(
        matriz, vetor_entrada, vetor_saida_colunas, tamanho_matriz, tamanho_matriz, repeticoes);

    double erro_maximo = CalcularErroMaximo(
        vetor_saida_linhas, vetor_saida_colunas, tamanho_matriz);

    fprintf(arquivo_saida, "%zu,%zu,%.12f,%.12f,%.12e\n",
            tamanho_matriz, repeticoes, tempo_linhas, tempo_colunas, erro_maximo);

    printf("N=%zu | linhas=%.6f s | colunas=%.6f s | erro=%.3e\n",
           tamanho_matriz, tempo_linhas, tempo_colunas, erro_maximo);

    free(matriz);
    free(vetor_entrada);
    free(vetor_saida_linhas);
    free(vetor_saida_colunas);
}

int main(void)
{
    const size_t tamanhos_matriz[] = {1000, 1500, 2000, 3000, 4000, 5000, 10000, 25000};

    const size_t quantidade_tamanhos =
        sizeof(tamanhos_matriz) / sizeof(tamanhos_matriz[0]);

    // arquivo pra plotar gráficos com python depois pro relatório
    FILE *arquivo_saida = fopen("resultados.csv", "w");
    if (arquivo_saida == NULL)
    {
        fprintf(stderr, "Erro ao criar resultados.csv.\n");
        return EXIT_FAILURE;
    }

    fprintf(arquivo_saida,
            "tamanho,repeticoes,tempo_linhas_s,tempo_colunas_s,erro_maximo\n");

    for (size_t indice_tamanho = 0; indice_tamanho < quantidade_tamanhos; ++indice_tamanho)
    {
        size_t n = tamanhos_matriz[indice_tamanho];

        int repeticoes;
        if (n <= 500)
        {
            repeticoes = 100;
        }
        else if (n <= 1500)
        {
            repeticoes = 20;
        }
        else
        {
            repeticoes = 5;
        }

        ExecutarTeste(repeticoes, n, arquivo_saida);
    }

    fclose(arquivo_saida);

    printf("\nResultados salvos em resultados.csv\n");

    return EXIT_SUCCESS;
}
