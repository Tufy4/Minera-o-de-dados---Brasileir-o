"""Gráficos da análise descritiva do Brasileirão.

Lê as tabelas geradas por brasileirao_pipeline.py, salva os gráficos em .png e abre cada
um numa janela (feche as janelas para encerrar o script).

Arquivos de entrada (na pasta de saída):
  ultimos10_por_temporada.csv   10 últimos colocados de cada temporada
  aproveitamento_resumo.csv     média, mediana, mínimo e máximo do aproveitamento
Saídas:
  desc_aproveitamento_temporada.png   média, mediana e faixa mín-máx por temporada
  desc_distribuicao_aproveitamento.png  histograma do aproveitamento dos 10 últimos
  desc_boxplot_temporada.png          distribuição do aproveitamento em cada temporada
  desc_gols_rebaixados.png            gols pró x gols contra (rebaixados x demais)
  desc_pontos_por_posicao.png         pontos médios de cada posição entre os 10 últimos
  desc_resultados_rebaixados.png      vitórias, empates e derrotas médias por grupo

Uso:  python analise_descritiva_visual.py [pasta_de_saida]
      (rode antes: python brasileirao_pipeline.py [pasta_de_saida])
Requer: matplotlib
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")  # sem tela (ex.: servidor): só salva os arquivos
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
N_REBAIXADOS = 4  # rebaixamento: 4 últimos colocados (vale também para 2003-2005 como aproximação)

AZUL, VERMELHO, CINZA = "#2a6fb0", "#c8402f", "#8a8f98"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100})


def ler(nome):
    if not (SAIDA / nome).exists():
        sys.exit(f"Arquivo {SAIDA / nome} não encontrado. Rode antes: python3 brasileirao_pipeline.py")
    with open(SAIDA / nome, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


ultimos = ler("ultimos10_por_temporada.csv")
resumo = ler("aproveitamento_resumo.csv")
for r in ultimos:
    for c in r:
        r[c] = float(r[c]) if c != "time" else r[c]
    r["temporada"] = int(r["temporada"]); r["posicao"] = int(r["posicao"])

# última posição de cada temporada = número de times; rebaixado = N últimos
n_times = {t: max(r["posicao"] for r in ultimos if r["temporada"] == t)
           for t in {r["temporada"] for r in ultimos}}
for r in ultimos:
    r["rebaixado"] = r["posicao"] > n_times[r["temporada"]] - N_REBAIXADOS

anos = sorted(n_times)
por_ano = {t: [r for r in ultimos if r["temporada"] == t] for t in anos}


def salvar(fig, nome):
    fig.savefig(SAIDA / nome, dpi=130, bbox_inches="tight")
    fig.canvas.manager.set_window_title(nome)


# 1) média, mediana e faixa por temporada
res_ano = [r for r in resumo if r["temporada"] != "GERAL"]
x = [int(r["temporada"]) for r in res_ano]
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(x, [float(r["minimo"]) for r in res_ano], [float(r["maximo"]) for r in res_ano],
                color=AZUL, alpha=.15, label="faixa mínimo–máximo")
ax.plot(x, [float(r["media"]) for r in res_ano], color=AZUL, marker="o", label="média")
ax.plot(x, [float(r["mediana"]) for r in res_ano], color=VERMELHO, marker="s", ls="--", label="mediana")
geral = next(r for r in resumo if r["temporada"] == "GERAL")
ax.axhline(float(geral["media"]), color=CINZA, ls=":", label=f"média geral ({geral['media']}%)")
ax.set(title="Aproveitamento dos 10 últimos colocados por temporada", xlabel="Temporada",
       ylabel="Aproveitamento (%)")
ax.legend(frameon=False)
salvar(fig, "desc_aproveitamento_temporada.png")

# 2) histograma do aproveitamento
todos = [r["aproveitamento"] for r in ultimos]
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(todos, bins=15, color=AZUL, alpha=.8, edgecolor="white")
ax.axvline(st.mean(todos), color=VERMELHO, label=f"média {st.mean(todos):.1f}%")
ax.axvline(st.median(todos), color="black", ls="--", label=f"mediana {st.median(todos):.1f}%")
ax.set(title="Distribuição do aproveitamento dos 10 últimos", xlabel="Aproveitamento (%)",
       ylabel="Times-temporada")
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.legend(frameon=False)
salvar(fig, "desc_distribuicao_aproveitamento.png")

# 3) boxplot por temporada
fig, ax = plt.subplots(figsize=(11, 5))
ax.boxplot([[r["aproveitamento"] for r in por_ano[t]] for t in anos], labels=anos,
           patch_artist=True, boxprops=dict(facecolor=AZUL, alpha=.4),
           medianprops=dict(color=VERMELHO))
ax.tick_params(axis="x", rotation=60)
ax.set(title="Aproveitamento dos 10 últimos: dispersão por temporada", xlabel="Temporada",
       ylabel="Aproveitamento (%)")
salvar(fig, "desc_boxplot_temporada.png")

# 4) gols pró x gols contra
fig, ax = plt.subplots(figsize=(7.5, 6))
for flag, cor, nome in ((False, CINZA, "salvos (5º ao 10º pior)"), (True, VERMELHO, "rebaixados")):
    pts = [r for r in ultimos if r["rebaixado"] == flag]
    ax.scatter([r["gols_pro"] for r in pts], [r["gols_contra"] for r in pts], c=cor, alpha=.7,
               label=f"{nome} (n={len(pts)})")
ax.set(title="Gols pró x gols contra entre os 10 últimos", xlabel="Gols pró",
       ylabel="Gols contra")
ax.legend(frameon=False)
salvar(fig, "desc_gols_rebaixados.png")

# 5) pontos médios por posição relativa (1 = último colocado)
pts_pos = defaultdict(list)
for r in ultimos:
    pts_pos[n_times[r["temporada"]] - r["posicao"] + 1].append(r["pontos"])
pos = sorted(pts_pos)
medias = [st.mean(pts_pos[p]) for p in pos]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([str(p) + "º" for p in pos], medias,
       color=[VERMELHO if p <= N_REBAIXADOS else AZUL for p in pos])
for i, m in enumerate(medias):
    ax.text(i, m + .5, f"{m:.1f}", ha="center", fontsize=9)
ax.set(title="Pontos médios por posição (1º = último colocado; vermelho = rebaixados)",
       xlabel="Posição contada a partir do último", ylabel="Pontos")
salvar(fig, "desc_pontos_por_posicao.png")

# 6) vitórias, empates, derrotas médias: rebaixados x demais
grupos = {"Rebaixados": [r for r in ultimos if r["rebaixado"]],
          "Salvos (5º ao 10º pior)": [r for r in ultimos if not r["rebaixado"]]}
fig, ax = plt.subplots(figsize=(7, 5))
base = [0, 0]
for campo, cor in (("vitorias", AZUL), ("empates", CINZA), ("derrotas", VERMELHO)):
    vals = [st.mean(r[campo] for r in g) for g in grupos.values()]
    ax.barh(list(grupos), vals, left=base, color=cor, label=campo)
    for i, v in enumerate(vals):
        ax.text(base[i] + v / 2, i, f"{v:.1f}", ha="center", va="center", color="white")
    base = [b + v for b, v in zip(base, vals)]
ax.set(title="Resultado médio por campanha", xlabel="Jogos")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
salvar(fig, "desc_resultados_rebaixados.png")


print("Gráficos salvos em:", SAIDA.resolve())
if matplotlib.get_backend().lower() != "agg":
    print("Abrindo os gráficos... feche as janelas para encerrar.")
    plt.show()
