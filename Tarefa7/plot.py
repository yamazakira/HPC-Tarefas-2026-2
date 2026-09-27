import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Gráfico 1 - Private, Firstprivate, Lastprivate, etc.
# ============================================================

df1 = pd.read_csv("resultados7.csv")

plt.figure(figsize=(12, 7))

for metodo in df1.columns[1:]:
    plt.plot(
        df1["numeros_verificados"],
        df1[metodo],
        marker="o",
        linewidth=2,
        label=metodo.replace("tempo_", "").replace("_", " ").title()
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Números verificados")
plt.ylabel("Tempo de execução (s)")
plt.title("Comparação das cláusulas de compartilhamento de variáveis")

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("tarefa71.png", dpi=300)
plt.show()


# ============================================================
# Gráfico 2 - Estratégias de distribuição de iterações
# ============================================================

df2 = pd.read_csv("resultados7esc.csv")

plt.figure(figsize=(12, 7))

for metodo in df2.columns[1:]:
    plt.plot(
        df2["numeros_verificados"],
        df2[metodo],
        marker="o",
        linewidth=2,
        label=metodo.replace("tempo_", "").title()
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Números verificados")
plt.ylabel("Tempo de execução (s)")
plt.title("Comparação das estratégias de distribuição de iterações")

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("tarefa72.png", dpi=300)
plt.show()