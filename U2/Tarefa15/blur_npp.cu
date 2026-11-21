/*
 * Tarefa 15 - Borramento (box blur) de imagens com NPP
 *
 * Usa a funcao de filtro de media (box filter) da NVIDIA Performance
 * Primitives: nppiFilterBoxBorder_8u_C{1,3,4}R_Ctx, com mascara
 * (2*RAIO+1) x (2*RAIO+1) e ancora no centro.
 *
 * Observacao sobre bordas: a versao sequencial calcula a media apenas
 * dos vizinhos que existem (janela recortada). O NPP nao possui esse
 * modo; usamos NPP_BORDER_REPLICATE (replica o pixel da borda). Assim,
 * o resultado e identico (a menos de arredondamento) nos pixels internos
 * e difere apenas na faixa de RAIO pixels junto as bordas.
 *
 * compilar: nvcc -O2 blur_npp.cu -o blur_npp -lnppif -lnppc
 * executar: ./blur_npp entrada.jpg saida.jpg
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <npp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RAIO_BLUR 25  // Janela de 51x51 pixels (igual a versao sequencial)
#define REPETICOES 5
#define AMOSTRAS_VERIFICACAO 2000
#define TOLERANCIA 1  // diferenca de arredondamento aceita (NPP arredonda)

#define ARQUIVO_JSON "resultados_npp.json"

#define CUDA_CHECK(chamada)                                              \
    do {                                                                 \
        cudaError_t erro_ = (chamada);                                   \
        if (erro_ != cudaSuccess) {                                      \
            fprintf(stderr, "Erro CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(erro_));                          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define NPP_CHECK(chamada)                                               \
    do {                                                                 \
        NppStatus status_ = (chamada);                                   \
        if (status_ != NPP_SUCCESS) {                                    \
            fprintf(stderr, "Erro NPP %s:%d: codigo %d\n", __FILE__,     \
                    __LINE__, (int)status_);                             \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/* --------------------------------------------------
 * Preenche o contexto de stream exigido pelas funcoes _Ctx do NPP
 * -------------------------------------------------- */
static NppStreamContext cria_contexto_npp(cudaStream_t stream)
{
    NppStreamContext ctx;
    memset(&ctx, 0, sizeof(ctx));

    int dispositivo = 0;
    CUDA_CHECK(cudaGetDevice(&dispositivo));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dispositivo));

    ctx.hStream = stream;
    ctx.nCudaDeviceId = dispositivo;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    CUDA_CHECK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor,
                                      cudaDevAttrComputeCapabilityMajor,
                                      dispositivo));
    CUDA_CHECK(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor,
                                      cudaDevAttrComputeCapabilityMinor,
                                      dispositivo));
    CUDA_CHECK(cudaStreamGetFlags(stream, &ctx.nStreamFlags));

    return ctx;
}

/* --------------------------------------------------
 * Chama o box filter do NPP de acordo com o numero de canais
 * -------------------------------------------------- */
static void filtro_npp(const Npp8u *d_entrada, int passo_entrada,
                       Npp8u *d_saida, int passo_saida,
                       int largura, int altura, int canais,
                       NppStreamContext ctx)
{
    NppiSize tamanho_img = {largura, altura};
    NppiPoint deslocamento = {0, 0};
    NppiSize mascara = {2 * RAIO_BLUR + 1, 2 * RAIO_BLUR + 1};
    NppiPoint ancora = {RAIO_BLUR, RAIO_BLUR};

    switch (canais) {
    case 1:
        NPP_CHECK(nppiFilterBoxBorder_8u_C1R_Ctx(
            d_entrada, passo_entrada, tamanho_img, deslocamento,
            d_saida, passo_saida, tamanho_img, mascara, ancora,
            NPP_BORDER_REPLICATE, ctx));
        break;
    case 3:
        NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(
            d_entrada, passo_entrada, tamanho_img, deslocamento,
            d_saida, passo_saida, tamanho_img, mascara, ancora,
            NPP_BORDER_REPLICATE, ctx));
        break;
    case 4:
        NPP_CHECK(nppiFilterBoxBorder_8u_C4R_Ctx(
            d_entrada, passo_entrada, tamanho_img, deslocamento,
            d_saida, passo_saida, tamanho_img, mascara, ancora,
            NPP_BORDER_REPLICATE, ctx));
        break;
    default:
        fprintf(stderr, "Numero de canais nao suportado: %d\n", canais);
        exit(EXIT_FAILURE);
    }
}

/* --------------------------------------------------
 * Referencia sequencial para um unico pixel/canal
 * -------------------------------------------------- */
static int media_referencia(const unsigned char *entrada, int x, int y, int c,
                            int largura, int altura, int canais)
{
    long soma = 0;
    int quantidade = 0;
    for (int dy = -RAIO_BLUR; dy <= RAIO_BLUR; dy++) {
        for (int dx = -RAIO_BLUR; dx <= RAIO_BLUR; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < largura && ny >= 0 && ny < altura) {
                soma += entrada[((size_t)ny * largura + nx) * canais + c];
                quantidade++;
            }
        }
    }
    return (int)(soma / quantidade);
}

/* --------------------------------------------------
 * Verifica pixels internos sorteados (fora da faixa de borda),
 * onde NPP e a versao sequencial devem coincidir.
 * -------------------------------------------------- */
static int verifica_amostras(const unsigned char *entrada,
                             const unsigned char *saida,
                             int largura, int altura, int canais,
                             int *maior_diferenca)
{
    int erros = 0;
    *maior_diferenca = 0;
    srand(42);
    for (int i = 0; i < AMOSTRAS_VERIFICACAO; i++) {
        int x = RAIO_BLUR + rand() % (largura - 2 * RAIO_BLUR);
        int y = RAIO_BLUR + rand() % (altura - 2 * RAIO_BLUR);
        for (int c = 0; c < canais; c++) {
            int ref = media_referencia(entrada, x, y, c,
                                       largura, altura, canais);
            int val = saida[((size_t)y * largura + x) * canais + c];
            int dif = abs(ref - val);
            if (dif > *maior_diferenca) *maior_diferenca = dif;
            if (dif > TOLERANCIA) erros++;
        }
    }
    return erros;
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

    size_t passo_host = (size_t)largura * canais;
    size_t tamanho = passo_host * altura;

    unsigned char *resultado = (unsigned char *)malloc(tamanho);

    if (resultado == NULL) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        stbi_image_free(imagem);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const NppLibraryVersion *versao = nppGetLibVersion();

    printf("GPU: %s | NPP %d.%d.%d\n", prop.name,
           versao->major, versao->minor, versao->build);
    printf("Imagem: %d x %d, canais: %d\n", largura, altura, canais);
    printf("Aplicando box filter NPP com mascara %dx%d...\n",
           2 * RAIO_BLUR + 1, 2 * RAIO_BLUR + 1);

    /* Memoria com pitch (linhas alinhadas), recomendado pelo NPP */
    Npp8u *d_entrada = NULL, *d_saida = NULL;
    size_t passo_entrada = 0, passo_saida = 0;

    CUDA_CHECK(cudaMallocPitch((void **)&d_entrada, &passo_entrada,
                               passo_host, altura));
    CUDA_CHECK(cudaMallocPitch((void **)&d_saida, &passo_saida,
                               passo_host, altura));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    NppStreamContext ctx = cria_contexto_npp(stream);

    cudaEvent_t ini, fim;
    CUDA_CHECK(cudaEventCreate(&ini));
    CUDA_CHECK(cudaEventCreate(&fim));

    /* ---------- Transferencia Host -> Device ---------- */
    float tempo_h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ini, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(d_entrada, passo_entrada, imagem, passo_host,
                                 passo_host, altura, cudaMemcpyHostToDevice,
                                 stream));
    CUDA_CHECK(cudaEventRecord(fim, stream));
    CUDA_CHECK(cudaEventSynchronize(fim));
    CUDA_CHECK(cudaEventElapsedTime(&tempo_h2d_ms, ini, fim));

    /* ---------- Aquecimento ---------- */
    filtro_npp(d_entrada, (int)passo_entrada, d_saida, (int)passo_saida,
               largura, altura, canais, ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* ---------- Filtro (REPETICOES vezes) ---------- */
    float soma_ms = 0.0f, min_ms = 1e30f, max_ms = 0.0f;

    for (int r = 0; r < REPETICOES; r++) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ini, stream));
        filtro_npp(d_entrada, (int)passo_entrada, d_saida, (int)passo_saida,
                   largura, altura, canais, ctx);
        CUDA_CHECK(cudaEventRecord(fim, stream));
        CUDA_CHECK(cudaEventSynchronize(fim));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ini, fim));
        soma_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    float media_ms = soma_ms / REPETICOES;

    /* ---------- Transferencia Device -> Host ---------- */
    float tempo_d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ini, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(resultado, passo_host, d_saida, passo_saida,
                                 passo_host, altura, cudaMemcpyDeviceToHost,
                                 stream));
    CUDA_CHECK(cudaEventRecord(fim, stream));
    CUDA_CHECK(cudaEventSynchronize(fim));
    CUDA_CHECK(cudaEventElapsedTime(&tempo_d2h_ms, ini, fim));

    double tempo_total_ms = tempo_h2d_ms + media_ms + tempo_d2h_ms;

    int maior_diferenca = 0;
    int erros = verifica_amostras(imagem, resultado, largura, altura, canais,
                                  &maior_diferenca);

    printf("[NPP] filtro: %9.3f ms (min %9.3f) | H2D: %.3f ms | "
           "D2H: %.3f ms | total: %9.3f ms\n",
           media_ms, min_ms, tempo_h2d_ms, tempo_d2h_ms, tempo_total_ms);
    printf("[NPP] verificacao (pixels internos): %d erros, maior diferenca %d\n",
           erros, maior_diferenca);

    FILE *json = fopen(ARQUIVO_JSON, "w");
    if (json != NULL) {
        fprintf(json,
                "{\n"
                "  \"versao\": \"npp\",\n"
                "  \"gpu\": \"%s\",\n"
                "  \"npp_versao\": \"%d.%d.%d\",\n"
                "  \"funcao\": \"nppiFilterBoxBorder_8u_C%dR_Ctx\",\n"
                "  \"borda\": \"NPP_BORDER_REPLICATE\",\n"
                "  \"largura\": %d,\n"
                "  \"altura\": %d,\n"
                "  \"canais\": %d,\n"
                "  \"raio\": %d,\n"
                "  \"repeticoes\": %d,\n"
                "  \"tempo_h2d_ms\": %f,\n"
                "  \"tempo_kernel_ms\": %f,\n"
                "  \"tempo_kernel_min_ms\": %f,\n"
                "  \"tempo_kernel_max_ms\": %f,\n"
                "  \"tempo_d2h_ms\": %f,\n"
                "  \"tempo_total_ms\": %f,\n"
                "  \"erros_verificacao\": %d,\n"
                "  \"maior_diferenca\": %d\n"
                "}\n",
                prop.name, versao->major, versao->minor, versao->build,
                canais, largura, altura, canais, RAIO_BLUR, REPETICOES,
                tempo_h2d_ms, media_ms, min_ms, max_ms, tempo_d2h_ms,
                tempo_total_ms, erros, maior_diferenca);
        fclose(json);
        printf("Resultados salvos em %s\n", ARQUIVO_JSON);
    } else {
        fprintf(stderr, "Erro ao abrir %s\n", ARQUIVO_JSON);
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
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_entrada));
    CUDA_CHECK(cudaFree(d_saida));
    stbi_image_free(imagem);
    free(resultado);

    return sucesso ? 0 : 1;
}
