"""
Tarefa 15 - Graficos do borramento de imagens (Sequencial x CUDA x NPP)

Le os arquivos JSON gerados na pasta da atividade:
    resultados_seq.json  -> blur.c       (sequencial)
    resultados_cuda.json -> blur_par.cu  (CUDA, varias grids/blocos)
    resultados_npp.json  -> blur_npp.cu  (NPP, box filter)

Gera na pasta da atividade:
    tarefa15_tempos.png    -> tempo de execucao de cada versao
    tarefa15_speedup.png   -> speedup em relacao a versao sequencial
    tarefa15_grid.png      -> tempo do kernel por organizacao de grid/bloco
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = os.path.dirname(os.path.abspath(__file__))


def carrega(nome):
    caminho = os.path.join(PASTA, nome)
    if not os.path.exists(caminho):
        print(f"[aviso] {nome} nao encontrado, ignorando.")
        return None
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


seq = carrega("resultados_seq.json")
cuda = carrega("resultados_cuda.json")
npp = carrega("resultados_npp.json")

if cuda is None and npp is None and seq is None:
    raise SystemExit("Nenhum resultado encontrado. Execute os programas antes.")


def rotulo_cfg(cfg):
    bx, by, _ = cfg["bloco"]
    if cfg["tipo"] == "1D":
        return f"1D\n{bx}"
    return f"{cfg['tipo']}\n{bx}x{by}"


# ============================================================
# Coleta dos tempos (em segundos)
# ============================================================
nomes, tempos_kernel, tempos_total = [], [], []

if seq is not None:
    nomes.append("Sequencial")
    tempos_kernel.append(seq["tempo_s"])
    tempos_total.append(seq["tempo_s"])

if cuda is not None:
    melhor = min(cuda["configuracoes"], key=lambda c: c["tempo_kernel_ms"])
    nomes.append("CUDA\n(" + rotulo_cfg(melhor).replace("\n", " ") + ")")
    tempos_kernel.append(melhor["tempo_kernel_ms"] / 1000)
    tempos_total.append(melhor["tempo_total_ms"] / 1000)

if npp is not None:
    nomes.append("NPP\n(FilterBoxBorder)")
    tempos_kernel.append(npp["tempo_kernel_ms"] / 1000)
    tempos_total.append(npp["tempo_total_ms"] / 1000)

info = seq or cuda or npp
titulo_img = (f"Imagem {info['largura']}x{info['altura']}x{info['canais']}, "
              f"janela {2 * info['raio'] + 1}x{2 * info['raio'] + 1}")
gpu = (cuda or npp or {}).get("gpu", "")

# ============================================================
# Grafico 1 - Tempos de execucao
# ============================================================
plt.figure(figsize=(10, 6))
pos = range(len(nomes))
largura_barra = 0.38

b1 = plt.bar([p - largura_barra / 2 for p in pos], tempos_kernel,
             largura_barra, label="Somente filtro/kernel")
b2 = plt.bar([p + largura_barra / 2 for p in pos], tempos_total,
             largura_barra, label="Com transferencias (H2D + D2H)")

for barras in (b1, b2):
    for barra in barras:
        altura = barra.get_height()
        plt.annotate(f"{altura:.4g} s",
                     (barra.get_x() + barra.get_width() / 2, altura),
                     ha="center", va="bottom", fontsize=8)

plt.yscale("log")
plt.xticks(list(pos), nomes)
plt.ylabel("Tempo de execução (s)")
plt.title(f"Borramento de imagem - tempo de execução\n{titulo_img}"
          + (f" | GPU: {gpu}" if gpu else ""))
plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, "tarefa15_tempos.png"), dpi=300)
plt.close()

# ============================================================
# Grafico 2 - Speedup em relacao ao sequencial
# ============================================================
if seq is not None and len(nomes) > 1:
    t_seq = seq["tempo_s"]
    plt.figure(figsize=(10, 6))
    nomes_par = nomes[1:]
    sp_kernel = [t_seq / t for t in tempos_kernel[1:]]
    sp_total = [t_seq / t for t in tempos_total[1:]]
    pos = range(len(nomes_par))

    b1 = plt.bar([p - largura_barra / 2 for p in pos], sp_kernel,
                 largura_barra, label="Somente filtro/kernel")
    b2 = plt.bar([p + largura_barra / 2 for p in pos], sp_total,
                 largura_barra, label="Com transferencias")

    for barras in (b1, b2):
        for barra in barras:
            altura = barra.get_height()
            plt.annotate(f"{altura:.0f}x",
                         (barra.get_x() + barra.get_width() / 2, altura),
                         ha="center", va="bottom", fontsize=9)

    plt.xticks(list(pos), nomes_par)
    plt.ylabel("Speedup (T_seq / T_par)")
    plt.title(f"Speedup em relação à versão sequencial\n{titulo_img}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA, "tarefa15_speedup.png"), dpi=300)
    plt.close()

# ============================================================
# Grafico 3 - Organizacao da grid (CUDA)
# ============================================================
if cuda is not None:
    cfgs = cuda["configuracoes"]
    rotulos = [rotulo_cfg(c) for c in cfgs]
    tempos = [c["tempo_kernel_ms"] for c in cfgs]
    ociosas = [c["threads_ociosas_pct"] for c in cfgs]
    cores = {"1D": "tab:gray", "2D": "tab:blue", "3D": "tab:green"}

    fig, ax1 = plt.subplots(figsize=(13, 7))
    barras = ax1.bar(range(len(cfgs)), tempos,
                     color=[cores[c["tipo"]] for c in cfgs])

    for barra, t in zip(barras, tempos):
        ax1.annotate(f"{t:.1f}", (barra.get_x() + barra.get_width() / 2, t),
                     ha="center", va="bottom", fontsize=8)

    if npp is not None:
        ax1.axhline(npp["tempo_kernel_ms"], color="tab:red", linestyle="--",
                    label=f"NPP ({npp['tempo_kernel_ms']:.1f} ms)")

    ax1.set_xticks(range(len(cfgs)))
    ax1.set_xticklabels(rotulos)
    ax1.set_xlabel("Tipo de grid e dimensões do bloco (threads)")
    ax1.set_ylabel("Tempo médio do kernel (ms)")
    ax1.set_ylim(0, max(tempos) * 1.35)

    ax2 = ax1.twinx()
    ax2.plot(range(len(cfgs)), ociosas, color="tab:orange", marker="o",
             linewidth=2, label="Threads ociosas (%)")
    ax2.set_ylabel("Threads ociosas nas bordas da grid (%)")
    ax2.set_ylim(0, max(max(ociosas) * 1.35, 1))

    from matplotlib.patches import Patch
    legenda = [Patch(color=v, label=f"Grid {k}") for k, v in cores.items()]
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(handles=legenda + h1 + h2, loc="upper center", ncol=5,
               fontsize=9)

    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.title(f"Impacto da organização da grid/blocos no kernel CUDA\n"
              f"{titulo_img} | GPU: {cuda['gpu']}")
    fig.tight_layout()
    plt.savefig(os.path.join(PASTA, "tarefa15_grid.png"), dpi=300)
    plt.close()

print("Graficos salvos em", PASTA)
