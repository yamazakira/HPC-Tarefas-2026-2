import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados.csv")

# Seleciona o maior tamanho de N
maior_n = df["tamanho_n"].max()
dados = df[df["tamanho_n"] == maior_n]

plt.figure(figsize=(10, 6))

plt.bar(
    dados["metodo"],
    dados["resultado"]
)

plt.axhline(
    3.141592653589793,
    linestyle="--",
    label="Valor esperado (π)"
)

plt.xlabel("Método")
plt.ylabel("Resultado")
plt.title(f"Resultado da integração para N = {maior_n:,}".replace(",", "."))

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("resultados_barras.png", dpi=300)
plt.show()