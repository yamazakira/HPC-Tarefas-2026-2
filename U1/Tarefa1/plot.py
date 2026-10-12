import pandas as pd
import matplotlib.pyplot as plt
import sys

script_name = sys.argv[0]
i = int(sys.argv[1])

df = pd.read_csv(f"Tarefa{i}/resultados.csv", comment="#")

plt.figure()
plt.plot(df["tamanho"], df["tempo_linhas_s"], marker="o", label="Acesso por linhas")
plt.plot(df["tamanho"], df["tempo_colunas_s"],marker="o", label="Acesso por colunas")
plt.xlabel("Tamanho da matriz (N x N)")
plt.ylabel("Tempo médio (s)")
plt.title(f"Multiplicação Matriz por Vetor - Tarefa {i}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"Graficos/grafico_{i}.png", dpi=150)
# plt.show()
