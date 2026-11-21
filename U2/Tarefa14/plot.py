import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_T13 = os.path.join(PASTA, "..", "Tarefa13")


def carrega(pasta, nome):
    caminho = os.path.join(pasta, nome)
    if not os.path.exists(caminho):
        print(f"[aviso] {caminho} nao encontrado")
        return None
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


uni = carrega(PASTA, "resultados_unificada.json")
exp = carrega(PASTA, "resultados_explicita.json")
omp_nao = carrega(PASTA_T13, "resultados_nao_otimizado.json")
omp_oti = carrega(PASTA_T13, "resultados_otimizado.json")


def ordena(pts):
    return sorted(pts, key=lambda r: r["nx"] * r["nz"])


p_uni = ordena(uni["resultados"])
p_exp = ordena(exp["resultados"])
n = [r["nx"] * r["nz"] for r in p_exp]
rotulos = [f"{r['nx']}x{r['nz']}" for r in p_exp]

plt.figure(figsize=(11, 7))
plt.plot(n, [r["tempo_total_s"] for r in p_uni], marker="o", linewidth=2,
         label="CUDA - memória unificada (cudaMallocManaged)")
plt.plot(n, [r["tempo_total_s"] for r in p_exp], marker="s", linewidth=2,
         label="CUDA - memória explícita (cudaMalloc + cudaMemcpy)")
if omp_nao:
    pts = ordena(omp_nao["resultados"])
    plt.plot([r["nx"] * r["nz"] for r in pts], [r["tempo_total_s"] for r in pts],
             marker="^", linestyle="--", label="OpenMP offload - não otimizado (Tarefa 13)")
if omp_oti:
    for modo, nome in (("gpu", "OpenMP offload - target data (Tarefa 13)"),
                       ("cpu", f"OpenMP CPU ({omp_oti['threads']} threads, Tarefa 13)")):
        pts = ordena([r for r in omp_oti["resultados"] if r["modo"] == modo])
        plt.plot([r["nx"] * r["nz"] for r in pts], [r["tempo_total_s"] for r in pts],
                 marker="v", linestyle="--", label=nome)
plt.xscale("log")
plt.yscale("log")
plt.xticks(n, rotulos)
plt.minorticks_off()
plt.xlabel("Tamanho da malha (NX x NZ)")
plt.ylabel(f"Tempo total ({exp['nt']} passos) (s)")
plt.title(f"Propagação da onda - CUDA x OpenMP offloading\nGPU: {exp['gpu']}")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa14_tempos.png"), dpi=300)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
h2d = np.array([r["tempo_h2d_s"] for r in p_exp])
d2h = np.array([r["tempo_d2h_s"] for r in p_exp])
laco = np.array([r["tempo_total_s"] for r in p_exp]) - h2d - d2h
ax1.bar(rotulos, laco, label="Laço temporal (kernels + 1 amostra/passo)")
ax1.bar(rotulos, h2d, bottom=laco, label="cudaMemcpy H2D (vel, u, u_old, u_new)")
ax1.bar(rotulos, d2h, bottom=laco + h2d, label="cudaMemcpy D2H (u final)")
ax1.set_yscale("log")
ax1.set_xlabel("Tamanho da malha")
ax1.set_ylabel("Tempo (s)")
ax1.set_title("Memória explícita - cálculo x transferências")
ax1.grid(True, axis="y", which="both", linestyle="--", alpha=0.5)
ax1.legend(fontsize=8)

razao = [a["tempo_total_s"] / b["tempo_total_s"] for a, b in zip(p_uni, p_exp)]
ax2.bar(rotulos, razao, color="tab:purple")
for i, v in enumerate(razao):
    ax2.annotate(f"{v:.2f}x", (i, v), ha="center", va="bottom")
ax2.axhline(1, color="black", linewidth=1)
ax2.set_xlabel("Tamanho da malha")
ax2.set_ylabel("Tempo unificada / tempo explícita")
ax2.set_title("Memória unificada x explícita")
ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa14_memoria.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
