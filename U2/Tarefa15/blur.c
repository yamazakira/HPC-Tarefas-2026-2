#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RAIO_BLUR 25  // Janela de 51x51 pixels

#define ARQUIVO_JSON "resultados_seq.json"

static double tempo_atual(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void blur_sequencial(
    const unsigned char *entrada,
    unsigned char *saida,
    int largura,
    int altura,
    int canais)
{
    for (int y = 0; y < altura; y++) {
        for (int x = 0; x < largura; x++) {
            for (int c = 0; c < canais; c++) {

                long soma = 0;
                int quantidade = 0;

                // Percorre a vizinhanca de cada pixel
                for (int dy = -RAIO_BLUR;
                     dy <= RAIO_BLUR; dy++) {

                    for (int dx = -RAIO_BLUR;
                         dx <= RAIO_BLUR; dx++) {

                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < largura &&
                            ny >= 0 && ny < altura) {

                            int indice =
                                (ny * largura + nx) * canais + c;

                            soma += entrada[indice];
                            quantidade++;
                        }
                    }
                }

                int indice_saida =
                    (y * largura + x) * canais + c;

                saida[indice_saida] =
                    (unsigned char)(soma / quantidade);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Uso: %s entrada.jpg saida.jpg\n", argv[0]);
        return 1;
    }

    int largura, altura, canais;

    // Leitura da imagem JPEG
    unsigned char *imagem = stbi_load(
        argv[1], &largura, &altura, &canais, 0);

    if (imagem == NULL) {
        fprintf(stderr, "Erro ao ler imagem: %s\n",
                stbi_failure_reason());
        return 1;
    }

    size_t tamanho =
        (size_t) largura * altura * canais;

    unsigned char *resultado = malloc(tamanho);

    if (resultado == NULL) {
        fprintf(stderr, "Erro ao alocar memoria.\n");
        stbi_image_free(imagem);
        return 1;
    }

    printf("Imagem: %d x %d, canais: %d\n",
           largura, altura, canais);
    printf("Aplicando blur com janela %dx%d...\n",
           2 * RAIO_BLUR + 1, 2 * RAIO_BLUR + 1);

    // Processamento sequencial (apenas o filtro e cronometrado)
    double t_ini = tempo_atual();

    blur_sequencial(
        imagem, resultado, largura, altura, canais);

    double t_fim = tempo_atual();
    double tempo_s = t_fim - t_ini;

    printf("[SEQ] Tempo do blur: %f s\n", tempo_s);

    // Exporta os resultados em JSON na pasta da atividade
    FILE *json = fopen(ARQUIVO_JSON, "w");
    if (json != NULL) {
        fprintf(json,
                "{\n"
                "  \"versao\": \"sequencial\",\n"
                "  \"largura\": %d,\n"
                "  \"altura\": %d,\n"
                "  \"canais\": %d,\n"
                "  \"raio\": %d,\n"
                "  \"tempo_s\": %f\n"
                "}\n",
                largura, altura, canais, RAIO_BLUR, tempo_s);
        fclose(json);
        printf("Resultados salvos em %s\n", ARQUIVO_JSON);
    } else {
        fprintf(stderr, "Erro ao abrir %s\n", ARQUIVO_JSON);
    }

    // Salva a imagem borrada em JPEG
    int sucesso = stbi_write_jpg(
        argv[2], largura, altura, canais, resultado, 95);

    if (!sucesso) {
        fprintf(stderr, "Erro ao salvar imagem.\n");
    } else {
        printf("Imagem borrada salva em: %s\n", argv[2]);
    }

    stbi_image_free(imagem);
    free(resultado);

    return sucesso ? 0 : 1;
}
// compilar gcc -O2 blur.c -o blur -lm
// executar ./blur entrada.jpg saida.jpg