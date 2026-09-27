import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados.csv")

plt.figure(figsize=(13, 8))

for metodo in df["metodo"].unique():
    dados = df[df["metodo"] == metodo].sort_values("tamanho_n")

    plt.plot(
        dados["tamanho_n"],
        dados["tempo"],
        marker="o",
        linewidth=2,
        label=metodo
    )

    # Mostra o tempo em cada ponto
    for _, linha in dados.iterrows():
        plt.annotate(
            f'{linha["tempo"]:.3f}s',
            (linha["tamanho_n"], linha["tempo"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7
        )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Número de subdivisões (N)")
plt.ylabel("Tempo de execução (s)")
plt.title("Tempo de execução dos métodos de integração")

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("resultados_trapezio2.png", dpi=300)
plt.show()