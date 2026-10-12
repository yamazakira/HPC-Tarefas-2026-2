import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados5.csv")

plt.figure(figsize=(10, 6))

plt.plot(
    df["numeros_verificados"],
    df["tempo_sequencial"],
    marker="o",
    label="Sequencial"
)

plt.plot(
    df["numeros_verificados"],
    df["tempo_paralelo"],
    marker="o",
    label="Paralelo"
)

plt.xlabel("Números verificados")
plt.ylabel("Tempo (segundos)")
plt.title("Tempo de execução: Sequencial x Paralelo")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("tarefa5.png", dpi=300)
plt.show()