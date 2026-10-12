from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

pasta = Path(__file__).resolve().parent

arquivos = {
    "O0": pasta / "resultadosOO.csv",
    "O2": pasta / "resultadosO2.csv",
    "O3": pasta / "resultadosO3.csv"
}

dados = {}

for nome, arquivo in arquivos.items():
    dados[nome] = pd.read_csv(arquivo)

metricas = {
    "inicializacao": "Tempo de inicialização",
    "soma_dependente": "Tempo da soma dependente",
    "soma_independente": "Tempo da soma independente"
}


def formatar_tempo(valor, pos):
    if valor >= 1:
        return f"{valor:.2f}"
    if valor >= 0.01:
        return f"{valor:.3f}"
    if valor >= 0.001:
        return f"{valor:.4f}"
    if valor >= 0.0001:
        return f"{valor:.5f}"
    return f"{valor:.6f}"


for coluna, titulo in metricas.items():
    plt.figure()

    for nome, df in dados.items():
        plt.plot(
            df["tamanho"],
            df[coluna],
            marker="o",
            label=nome
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(formatar_tempo)
    )

    plt.xlabel("Tamanho do vetor")
    plt.ylabel("Tempo (s)")
    plt.title(titulo)

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()

    arquivo_saida = pasta / f"{coluna}.png"
    plt.savefig(arquivo_saida, dpi=300)
    plt.close()

    print(f"Gráfico salvo em: {arquivo_saida}")
