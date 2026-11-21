/*
 * Tarefa 15 - Borramento (box blur) de imagens em CUDA
 *
 * Versao paralela que explora grids multidimensionais:
 *   - GRID 1D : um indice linear por pixel (referencia para comparacao)
 *   - GRID 2D : threads mapeadas diretamente em (x, y) da imagem
 *   - GRID 3D : (x, y) da imagem + canal de cor no eixo z da grid
 *
 * Para cada configuracao de bloco o kernel e executado REPETICOES vezes
 * e os tempos (medidos com cudaEvent) sao exportados em JSON.
 *
 * compilar: nvcc -O2 blur_par.cu -o blur_par
 * executar: ./blur_par entrada.jpg saida.jpg
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RAIO_BLUR 25  // Janela de 51x51 pixels (igual a versao sequencial)
#define REPETICOES 5
#define AMOSTRAS_VERIFICACAO 2000

#define ARQUIVO_JSON "resultados_cuda.json"

#define CUDA_CHECK(chamada)                                              \
    do {                                                                 \
        cudaError_t erro_ = (chamada);                                   \
        if (erro_ != cudaSuccess) {                                      \
            fprintf(stderr, "Erro CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(erro_));                          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* --------------------------------------------------
 * Configuracoes de grid/bloco testadas
 * -------------------------------------------------- */
typedef enum { GRID_1D = 1, GRID_2D = 2, GRID_3D = 3 } TipoGrid;

typedef struct {
    TipoGrid tipo;
    unsigned int bx;
    unsigned int by;
} Configuracao;

static const Configuracao configuracoes[] = {
    {GRID_1D, 256, 1},
    {GRID_1D, 1024, 1},
    {GRID_2D, 8, 8},
    {GRID_2D, 16, 16},
    {GRID_2D, 32, 32},
    {GRID_2D, 32, 8},
    {GRID_2D, 64, 4},
    {GRID_2D, 128, 1},
    {GRID_2D, 8, 32},
    {GRID_2D, 1, 128},
    {GRID_3D, 16, 16},
    {GRID_3D, 32, 8},
};

static const size_t quantidade_configuracoes =
    sizeof(configuracoes) / sizeof(configuracoes[0]);

/* --------------------------------------------------
 * Calcula a media da vizinhanca de um pixel/canal.
 * Os limites da janela sao recortados na borda da imagem,
 * o que equivale ao "if" da versao sequencial.
 * -------------------------------------------------- */
__host__ __device__ inline unsigned char media_vizinhanca(
    const unsigned char *entrada, int x, int y, int c,
    int largura, int altura, int canais)
{
    int x0 = x - RAIO_BLUR < 0 ? 0 : x - RAIO_BLUR;
    int y0 = y - RAIO_BLUR < 0 ? 0 : y - RAIO_BLUR;
    int x1 = x + RAIO_BLUR >= largura ? largura - 1 : x + RAIO_BLUR;
    int y1 = y + RAIO_BLUR >= altura ? altura - 1 : y + RAIO_BLUR;

    long soma = 0;
    for (int ny = y0; ny <= y1; ny++) {
        const unsigned char *linha = entrada + (size_t)ny * largura * canais;
        for (int nx = x0; nx <= x1; nx++) {
            soma += linha[nx * canais + c];
        }
    }

    int quantidade = (x1 - x0 + 1) * (y1 - y0 + 1);
    return (unsigned char)(soma / quantidade);
}

/* --------------------------------------------------
 * GRID 1D: cada thread recebe um indice linear de pixel
 * e precisa reconstruir (x, y) com divisao/modulo.
 * -------------------------------------------------- */
__global__ void blur_kernel_1d(const unsigned char *entrada,
                               unsigned char *saida,
                               int largura, int altura, int canais)
{
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)largura * altura;

    if (id >= total) return;

    int x = (int)(id % largura);
    int y = (int)(id / largura);

    for (int c = 0; c < canais; c++) {
        saida[id * canais + c] =
            media_vizinhanca(entrada, x, y, c, largura, altura, canais);
    }
}

/* --------------------------------------------------
 * GRID 2D: o par (blockIdx, threadIdx) em x e y corresponde
 * diretamente a coluna e linha do pixel.
 * -------------------------------------------------- */
__global__ void blur_kernel_2d(const unsigned char *entrada,
                               unsigned char *saida,
                               int largura, int altura, int canais)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= largura || y >= altura) return;

    size_t indice = ((size_t)y * largura + x) * canais;

    for (int c = 0; c < canais; c++) {
        saida[indice + c] =
            media_vizinhanca(entrada, x, y, c, largura, altura, canais);
    }
}

/* --------------------------------------------------
 * GRID 3D: alem de (x, y), o eixo z da grid indexa o canal.
 * Cada thread calcula um unico valor (pixel, canal).
 * -------------------------------------------------- */
__global__ void blur_kernel_3d(const unsigned char *entrada,
                               unsigned char *saida,
                               int largura, int altura, int canais)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= largura || y >= altura || c >= canais) return;

    size_t indice = ((size_t)y * largura + x) * canais + c;

    saida[indice] =
        media_vizinhanca(entrada, x, y, c, largura, altura, canais);
}

/* --------------------------------------------------
 * Monta a grid de acordo com o tipo e lanca o kernel
 * -------------------------------------------------- */
static void monta_grid(const Configuracao *cfg, int largura, int altura,
                       int canais, dim3 *grid, dim3 *bloco)
{
    if (cfg->tipo == GRID_1D) {
        size_t total = (size_t)largura * altura;
        *bloco = dim3(cfg->bx, 1, 1);
        *grid = dim3((unsigned int)((total + cfg->bx - 1) / cfg->bx), 1, 1);
    } else {
        *bloco = dim3(cfg->bx, cfg->by, 1);
        *grid = dim3((largura + cfg->bx - 1) / cfg->bx,
                     (altura + cfg->by - 1) / cfg->by,
                     cfg->tipo == GRID_3D ? canais : 1);
    }
}

static void lanca_kernel(const Configuracao *cfg, dim3 grid, dim3 bloco,
                         const unsigned char *d_entrada,
                         unsigned char *d_saida,
                         int largura, int altura, int canais)
{
    switch (cfg->tipo) {
    case GRID_1D:
        blur_kernel_1d<<<grid, bloco>>>(d_entrada, d_saida,
                                        largura, altura, canais);
        break;
    case GRID_2D:
        blur_kernel_2d<<<grid, bloco>>>(d_entrada, d_saida,
                                        largura, altura, canais);
        break;
    case GRID_3D:
        blur_kernel_3d<<<grid, bloco>>>(d_entrada, d_saida,
                                        largura, altura, canais);
        break;
    }
}

/* --------------------------------------------------
 * Verifica o resultado da GPU contra a referencia sequencial
 * em pixels sorteados (a versao sequencial completa e lenta).
 * Retorna o numero de divergencias.
 * -------------------------------------------------- */
static int verifica_amostras(const unsigned char *entrada,
                             const unsigned char *saida_gpu,
                             int largura, int altura, int canais)
{
    int erros = 0;
    srand(42);
    for (int i = 0; i < AMOSTRAS_VERIFICACAO; i++) {
        int x = rand() % largura;
        int y = rand() % altura;
        for (int c = 0; c < canais; c++) {
            unsigned char ref =
                media_vizinhanca(entrada, x, y, c, largura, altura, canais);
            size_t indice = ((size_t)y * largura + x) * canais + c;
            if (ref != saida_gpu[indice]) erros++;
        }
    }
    return erros;
}

static const char *nome_tipo(TipoGrid tipo)
{
    switch (tipo) {
    case GRID_1D: return "1D";
    case GRID_2D: return "2D";
    case GRID_3D: return "3D";
    }
    return "?";
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Uso: %s entrada.jpg saida.jpg\n", argv[0]);
        return 1;
    }

    int largura, altura, canais;

    unsigned char *imagem = stbi_load(argv[1], &largura, &altura, &canais, 0);

    if (imagem == NULL) {
        fprintf(stderr, "Erro ao ler imagem: %s\n", stbi_failure_reason());
        return 1;
    }

    size_t tamanho = (size_t)largura * altura * canais;

    unsigned char *resultado = (unsigned char *)malloc(tamanho);

    if (resultado == NULL) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        stbi_image_free(imagem);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s (SMs: %d)\n", prop.name, prop.multiProcessorCount);
    printf("Imagem: %d x %d, canais: %d\n", largura, altura, canais);
    printf("Aplicando blur com janela %dx%d...\n",
           2 * RAIO_BLUR + 1, 2 * RAIO_BLUR + 1);

    unsigned char *d_entrada = NULL;
    unsigned char *d_saida = NULL;

    CUDA_CHECK(cudaMalloc(&d_entrada, tamanho));
    CUDA_CHECK(cudaMalloc(&d_saida, tamanho));

    cudaEvent_t ini, fim;
    CUDA_CHECK(cudaEventCreate(&ini));
    CUDA_CHECK(cudaEventCreate(&fim));

    /* ---------- Transferencia Host -> Device ---------- */
    float tempo_h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ini));
    CUDA_CHECK(cudaMemcpy(d_entrada, imagem, tamanho, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(fim));
    CUDA_CHECK(cudaEventSynchronize(fim));
    CUDA_CHECK(cudaEventElapsedTime(&tempo_h2d_ms, ini, fim));

    /* ---------- Aquecimento (inicializacao do contexto/kernel) ---------- */
    {
        dim3 grid, bloco;
        Configuracao aquecimento = {GRID_2D, 16, 16};
        monta_grid(&aquecimento, largura, altura, canais, &grid, &bloco);
        lanca_kernel(&aquecimento, grid, bloco, d_entrada, d_saida,
                     largura, altura, canais);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    FILE *json = fopen(ARQUIVO_JSON, "w");
    if (json == NULL) {
        fprintf(stderr, "Erro ao abrir %s\n", ARQUIVO_JSON);
        return 1;
    }

    fprintf(json, "{\n");
    fprintf(json, "  \"versao\": \"cuda\",\n");
    fprintf(json, "  \"gpu\": \"%s\",\n", prop.name);
    fprintf(json, "  \"largura\": %d,\n", largura);
    fprintf(json, "  \"altura\": %d,\n", altura);
    fprintf(json, "  \"canais\": %d,\n", canais);
    fprintf(json, "  \"raio\": %d,\n", RAIO_BLUR);
    fprintf(json, "  \"repeticoes\": %d,\n", REPETICOES);
    fprintf(json, "  \"tempo_h2d_ms\": %f,\n", tempo_h2d_ms);

    float tempo_d2h_ms = 0.0f;
    float melhor_kernel_ms = 1e30f;

    fprintf(json, "  \"configuracoes\": [\n");

    for (size_t i = 0; i < quantidade_configuracoes; i++) {
        const Configuracao *cfg = &configuracoes[i];

        dim3 grid, bloco;
        monta_grid(cfg, largura, altura, canais, &grid, &bloco);

        CUDA_CHECK(cudaMemset(d_saida, 0, tamanho));

        float soma_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;

        for (int r = 0; r < REPETICOES; r++) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventRecord(ini));
            lanca_kernel(cfg, grid, bloco, d_entrada, d_saida,
                         largura, altura, canais);
            CUDA_CHECK(cudaEventRecord(fim));
            CUDA_CHECK(cudaEventSynchronize(fim));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventElapsedTime(&ms, ini, fim));
            soma_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }

        float media_ms = soma_ms / REPETICOES;

        /* ---------- Transferencia Device -> Host ---------- */
        float ms_d2h = 0.0f;
        CUDA_CHECK(cudaEventRecord(ini));
        CUDA_CHECK(cudaMemcpy(resultado, d_saida, tamanho,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(fim));
        CUDA_CHECK(cudaEventSynchronize(fim));
        CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ini, fim));
        tempo_d2h_ms = ms_d2h;

        int erros = verifica_amostras(imagem, resultado,
                                      largura, altura, canais);

        /* Threads lancadas x threads uteis (desperdicio nas bordas) */
        unsigned long long threads_lancadas =
            (unsigned long long)grid.x * grid.y * grid.z *
            bloco.x * bloco.y * bloco.z;
        unsigned long long threads_uteis =
            (unsigned long long)largura * altura *
            (cfg->tipo == GRID_3D ? canais : 1);
        double ociosas_pct =
            100.0 * (double)(threads_lancadas - threads_uteis) /
            (double)threads_lancadas;

        double tempo_total_ms = tempo_h2d_ms + media_ms + ms_d2h;

        printf("[CUDA %s] bloco (%3u,%3u,%u) grid (%5u,%5u,%u) | "
               "kernel: %9.3f ms (min %9.3f) | total: %9.3f ms | "
               "ociosas: %5.2f%% | erros: %d\n",
               nome_tipo(cfg->tipo), bloco.x, bloco.y, bloco.z,
               grid.x, grid.y, grid.z, media_ms, min_ms, tempo_total_ms,
               ociosas_pct, erros);

        fprintf(json,
                "    {\"tipo\": \"%s\", \"bloco\": [%u, %u, %u], "
                "\"grid\": [%u, %u, %u], "
                "\"threads_por_bloco\": %u, "
                "\"threads_lancadas\": %llu, "
                "\"threads_ociosas_pct\": %f, "
                "\"tempo_kernel_ms\": %f, \"tempo_kernel_min_ms\": %f, "
                "\"tempo_kernel_max_ms\": %f, \"tempo_d2h_ms\": %f, "
                "\"tempo_total_ms\": %f, \"erros_verificacao\": %d}%s\n",
                nome_tipo(cfg->tipo), bloco.x, bloco.y, bloco.z,
                grid.x, grid.y, grid.z, bloco.x * bloco.y * bloco.z,
                threads_lancadas, ociosas_pct, media_ms, min_ms, max_ms,
                ms_d2h, tempo_total_ms, erros,
                i + 1 < quantidade_configuracoes ? "," : "");

        if (media_ms < melhor_kernel_ms) melhor_kernel_ms = media_ms;
    }

    fprintf(json, "  ],\n");
    fprintf(json, "  \"tempo_d2h_ms\": %f,\n", tempo_d2h_ms);
    fprintf(json, "  \"melhor_tempo_kernel_ms\": %f\n", melhor_kernel_ms);
    fprintf(json, "}\n");
    fclose(json);

    printf("Resultados salvos em %s\n", ARQUIVO_JSON);

    /* Salva a imagem gerada pela configuracao 2D 16x16 */
    {
        dim3 grid, bloco;
        Configuracao cfg = {GRID_2D, 16, 16};
        monta_grid(&cfg, largura, altura, canais, &grid, &bloco);
        lanca_kernel(&cfg, grid, bloco, d_entrada, d_saida,
                     largura, altura, canais);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(resultado, d_saida, tamanho,
                              cudaMemcpyDeviceToHost));
    }

    int sucesso = stbi_write_jpg(argv[2], largura, altura, canais,
                                 resultado, 95);

    if (!sucesso) {
        fprintf(stderr, "Erro ao salvar imagem.\n");
    } else {
        printf("Imagem borrada salva em: %s\n", argv[2]);
    }

    CUDA_CHECK(cudaEventDestroy(ini));
    CUDA_CHECK(cudaEventDestroy(fim));
    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_saida));
    stbi_image_free(imagem);
    free(resultado);

    return sucesso ? 0 : 1;
}
