import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))


def carrega(nome):
    with open(os.path.join(PASTA, nome), encoding="utf-8") as f:
        return json.load(f)


nao = carrega("resultados_nao_otimizado.json")
oti = carrega("resultados_otimizado.json")
nt = nao["nt"]
dispositivo = "GPU" if oti["dispositivos"] > 0 else "host (sem GPU)"


def serie(dados, modo):
    pts = sorted((r for r in dados["resultados"] if r["modo"] == modo),
                 key=lambda r: r["nx"] * r["nz"])
    return [r["nx"] * r["nz"] for r in pts], pts


rotulo_tam = [f"{r['nx']}x{r['nz']}" for r in serie(nao, "gpu_map")[1]]

plt.figure(figsize=(11, 7))
n, pts = serie(nao, "gpu_map")
plt.plot(n, [p["tempo_total_s"] for p in pts], marker="o", linewidth=2,
         label="Não otimizado (map a cada passo)")
for modo, nome in (("gpu", "Otimizado - target data (sempre GPU)"),
                   ("cpu", f"Otimizado - if(false): CPU ({oti['threads']} threads)"),
                   ("auto", f"Otimizado - automático (GPU se pontos ≥ {oti['limiar_gpu']})")):
    n, pts = serie(oti, modo)
    plt.plot(n, [p["tempo_total_s"] for p in pts], marker="s",
             linestyle="--" if modo == "auto" else "-", linewidth=2, label=nome)
plt.axvline(oti["limiar_gpu"], color="gray", linestyle=":")
plt.xscale("log")
plt.yscale("log")
plt.xticks(n, rotulo_tam)
plt.minorticks_off()
plt.xlabel("Tamanho da malha (NX x NZ)")
plt.ylabel(f"Tempo total ({nt} passos) (s)")
plt.title(f"Propagação da onda - OpenMP offloading ({dispositivo})")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa13_tempos.png"), dpi=300)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
n, p_nao = serie(nao, "gpu_map")
_, p_gpu = serie(oti, "gpu")
_, p_cpu = serie(oti, "cpu")
x = np.arange(len(n))
ax1.bar(x - 0.2, [a["tempo_total_s"] / b["tempo_total_s"] for a, b in zip(p_nao, p_gpu)],
        0.4, label="Não otimizado / otimizado (GPU)")
ax1.bar(x + 0.2, [c["tempo_total_s"] / b["tempo_total_s"] for c, b in zip(p_cpu, p_gpu)],
        0.4, label="CPU / otimizado (GPU)")
ax1.axhline(1, color="black", linewidth=1)
ax1.set_xticks(x)
ax1.set_xticklabels(rotulo_tam)
ax1.set_xlabel("Tamanho da malha")
ax1.set_ylabel("Speedup da versão otimizada na GPU")
ax1.set_title("Ganho do target data e vantagem da GPU")
ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
ax1.legend()

volume_nao = [nt * 5 * k * 8 / 1e9 for k in n]
volume_oti = [(4 + 1) * k * 8 / 1e9 + nt * 8 / 1e9 for k in n]
ax2.bar(x - 0.2, volume_nao, 0.4, label="Não otimizado: 4 vetores ida + u_new volta, por passo")
ax2.bar(x + 0.2, volume_oti, 0.4, label="Otimizado: 4 vetores ida + u volta, uma vez")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(rotulo_tam)
ax2.set_xlabel("Tamanho da malha")
ax2.set_ylabel("Volume transferido CPU ↔ GPU (GB)")
ax2.set_title("Transferências de dados")
ax2.grid(True, axis="y", which="both", linestyle="--", alpha=0.5)
ax2.legend(fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa13_otimizacao.png"), dpi=300)
plt.close()

plt.figure(figsize=(11, 6))
h2d = [p["tempo_h2d_s"] for p in p_gpu]
d2h = [p["tempo_d2h_s"] for p in p_gpu]
laco = [p["tempo_total_s"] - a - b for p, a, b in zip(p_gpu, h2d, d2h)]
plt.bar(rotulo_tam, laco, label="Laço temporal na GPU")
plt.bar(rotulo_tam, h2d, bottom=laco, label="target enter data (H2D)")
plt.bar(rotulo_tam, d2h, bottom=[a + b for a, b in zip(laco, h2d)], label="target exit data (D2H)")
plt.yscale("log")
plt.xlabel("Tamanho da malha")
plt.ylabel("Tempo (s)")
plt.title("Versão otimizada - cálculo x transferências")
plt.grid(True, axis="y", which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa13_transferencias.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
