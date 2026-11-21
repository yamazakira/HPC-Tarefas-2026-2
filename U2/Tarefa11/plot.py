import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PASTA, "resultados.json"), encoding="utf-8") as f:
    dados = json.load(f)

res = dados["resultados"]
n = [r["tamanho"] for r in res]
dispositivo = "GPU" if dados["dispositivos"] > 0 else "host (sem GPU)"

plt.figure(figsize=(11, 7))
plt.plot(n, [r["cpu_s"] for r in res], marker="o", linewidth=2,
         label=f"OpenMP CPU ({dados['threads']} threads)")
plt.plot(n, [r["gpu_total_s"] for r in res], marker="s", linewidth=2,
         label=f"OpenMP offload - total com transferências ({dispositivo})")
plt.plot(n, [r["gpu_kernel_s"] for r in res], marker="^", linestyle="--",
         label="OpenMP offload - apenas cálculo")
plt.plot(n, [r["gpu_transferencia_s"] for r in res], marker="v", linestyle=":",
         label="OpenMP offload - apenas transferências (B, C → GPU; A → CPU)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Tamanho do vetor")
plt.ylabel("Tempo de execução (s)")
plt.title("A = B + C - CPU x GPU (OpenMP)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa11_tempos.png"), dpi=300)
plt.close()

plt.figure(figsize=(11, 6))
fracao = [100 * r["gpu_transferencia_s"] /
          (r["gpu_transferencia_s"] + r["gpu_kernel_s"]) for r in res]
plt.bar([f"{t:.0e}" for t in n], fracao, color="tab:orange")
for i, v in enumerate(fracao):
    plt.annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom")
plt.ylim(0, 105)
plt.xlabel("Tamanho do vetor")
plt.ylabel("Parcela do tempo da GPU gasta em transferências (%)")
plt.title("Impacto da transferência de dados CPU ↔ GPU")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa11_transferencia.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
