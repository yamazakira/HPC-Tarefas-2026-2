"""
Tarefa 17 - Graficos do ping-pong MPI (MPI_Send, MPI_Bsend, MPI_Rsend, MPI_Ssend)

Le resultados_{send,bsend,rsend,ssend}.json e salva na pasta da atividade:
    tarefa17_tempo.png  -> tempo de um sentido x tamanho da mensagem (log-log),
                           com o modelo t(n) = alfa + n/beta e os regimes
                           dominados por latencia e por largura de banda
    tarefa17_banda.png  -> largura de banda efetiva x tamanho da mensagem
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))

MODOS = [
    ("send", "MPI_Send (padrão)"),
    ("bsend", "MPI_Bsend (bufferizado)"),
    ("rsend", "MPI_Rsend (pronto)"),
    ("ssend", "MPI_Ssend (síncrono)"),
]

dados = {}
for modo, _ in MODOS:
    caminho = os.path.join(PASTA, f"resultados_{modo}.json")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            dados[modo] = json.load(f)
    else:
        print(f"[aviso] resultados_{modo}.json nao encontrado")

if not dados:
    raise SystemExit("Nenhum resultado encontrado.")

ref = dados.get("send", next(iter(dados.values())))
subtitulo = (f"{ref['comunicacao']} ({ref['no_processo_0']} ↔ "
             f"{ref['no_processo_1']})")

# ------------------------------------------------------------
# Modelo de Hockney: t(n) = alfa + n / beta (ajustado no MPI_Send)
#   alfa: latencia (tempo da menor mensagem)
#   beta: largura de banda assintotica (maior banda medida)
#   n* = alfa * beta: tamanho em que os dois termos se igualam
# ------------------------------------------------------------
res_ref = ref["resultados"]
bytes_ref = np.array([r["bytes"] for r in res_ref], dtype=float)
lat_ref = np.array([r["latencia_us"] for r in res_ref]) * 1e-6
alfa = float(np.mean(lat_ref[:3]))
beta = float(max(r["banda_mbs"] for r in res_ref) * 1e6)
n_critico = alfa * beta

print(f"Latencia (alfa) ~ {alfa * 1e6:.2f} us")
print(f"Banda assintotica (beta) ~ {beta / 1e6:.1f} MB/s")
print(f"Tamanho critico n* = alfa*beta ~ {n_critico:.0f} bytes")


def formata_bytes(n):
    for unidade in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unidade}"
        n /= 1024
    return f"{n:.0f} TB"


# ============================================================
# Grafico 1 - Tempo x tamanho
# ============================================================
plt.figure(figsize=(12, 7))
for modo, nome in MODOS:
    if modo not in dados:
        continue
    res = dados[modo]["resultados"]
    plt.plot([r["bytes"] for r in res], [r["latencia_us"] for r in res],
             marker="o", linewidth=2, label=nome)

xs = np.logspace(np.log10(bytes_ref.min()), np.log10(bytes_ref.max()), 200)
plt.plot(xs, (alfa + xs / beta) * 1e6, "k--", linewidth=1,
         label=f"Modelo α + n/β  (α={alfa * 1e6:.2f} µs, β={beta / 1e9:.2f} GB/s)")

plt.axvline(n_critico, color="gray", linestyle=":")
plt.axvspan(bytes_ref.min() / 1.5, n_critico, color="tab:orange", alpha=0.08)
plt.axvspan(n_critico, bytes_ref.max() * 1.5, color="tab:blue", alpha=0.08)
ymax = max(r["latencia_us"] for d in dados.values() for r in d["resultados"])
plt.text(bytes_ref.min() * 1.2, ymax * 0.5, "Latência domina",
         color="tab:orange", fontsize=11)
plt.text(n_critico * 1.5, ymax * 0.5, "Largura de banda domina",
         color="tab:blue", fontsize=11)
plt.text(n_critico * 1.1, (alfa * 1e6) * 0.6, f"n* ≈ {formata_bytes(n_critico)}",
         color="gray", fontsize=9)

plt.xscale("log", base=2)
plt.yscale("log")
plt.xlim(bytes_ref.min() / 1.5, bytes_ref.max() * 1.5)
plt.xlabel("Tamanho da mensagem (bytes)")
plt.ylabel("Tempo de um sentido (µs)  [ida-e-volta / 2]")
plt.title("Ping-pong MPI - tempo x tamanho da mensagem\n" + subtitulo)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa17_tempo.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Largura de banda x tamanho
# ============================================================
plt.figure(figsize=(12, 7))
for modo, nome in MODOS:
    if modo not in dados:
        continue
    res = dados[modo]["resultados"]
    plt.plot([r["bytes"] for r in res], [r["banda_mbs"] for r in res],
             marker="o", linewidth=2, label=nome)

plt.plot(xs, xs / (alfa + xs / beta) / 1e6, "k--", linewidth=1,
         label="Modelo n / (α + n/β)")
plt.axvline(n_critico, color="gray", linestyle=":",
            label=f"n* ≈ {formata_bytes(n_critico)} (metade de β)")

plt.xscale("log", base=2)
plt.xlabel("Tamanho da mensagem (bytes)")
plt.ylabel("Largura de banda efetiva (MB/s)")
plt.title("Ping-pong MPI - largura de banda x tamanho da mensagem\n" + subtitulo)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa17_banda.png"), dpi=300)
plt.close()

print("Graficos salvos em", PASTA)
