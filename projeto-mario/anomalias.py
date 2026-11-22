"""Detecção de anomalias no rebaixamento: campanhas da parte de baixo da tabela que fogem do
padrão histórico (2006-2025).

Campanhas analisadas: do 13º ao 20º colocado de cada temporada (160 campanhas)
  rebaixados            17º ao 20º
  escaparam por pouco   13º ao 16º

Método 1 - pontuação x linha de corte das outras temporadas (estatístico, não paramétrico)
  A linha de corte de uma temporada é a pontuação do 17º colocado (o melhor dos rebaixados):
  para escapar, era preciso fazer mais pontos que ele. Para cada campanha, conta em quantas das
  outras 19 temporadas aquela pontuação teria dado outro destino:
    rebaixado: % das outras temporadas em que esses pontos SALVARIAM o time
    escapou:   % das outras temporadas em que esses pontos DERRUBARIAM o time
  Empate com a linha conta como "cairia" (dependeria do critério de desempate).
  Anomalia quando o destino seria outro em pelo menos 75% das temporadas:
    "caiu com muitos pontos"  e  "escapou com poucos pontos"

Método 2 - Isolation Forest (multivariado)
  Usa aproveitamento, % de vitórias, % de empates, gols pró e gols contra por jogo e marca as
  ~10% campanhas mais fáceis de isolar, ou seja, com o perfil mais diferente das outras. A coluna
  "motivo_if" mostra a variável mais fora do normal em cada uma (z-score).

Punição do STJD em 2013 (Portuguesa e Flamengo perderam 4 pontos) aplicada: sem ela, a tabela
calculada pelos placares rebaixaria o Fluminense com 46 pontos, e ele seria a maior "anomalia"
da base sem ter caído de verdade.

Arquivos de entrada (na mesma pasta do script):
  brasileirao_limpo.csv   histórico limpo (gerado por brasileirao_pipeline.py)
Saídas:
  anomalias_campanhas.csv     as 160 campanhas com as medidas dos dois métodos
  anomalias_rebaixamento.png  pontos dos 8 últimos por temporada, com as anomalias destacadas

Uso:  python anomalias.py [pasta_de_saida]
Requer: pandas, scikit-learn, matplotlib
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

N_REBAIXADOS = 4
PASTA = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)

LIMITE_OUTRO_DESTINO = 0.75   # método 1: outro destino em pelo menos 3 de cada 4 temporadas
CONTAMINACAO = 0.10           # método 2: fração de campanhas marcadas pelo Isolation Forest
FEATURES = ["aproveitamento", "pct_vitorias", "pct_empates", "gols_pro_jogo", "gols_contra_jogo"]
NOME_VAR = {"aproveitamento": "aproveitamento", "pct_vitorias": "% de vitórias",
            "pct_empates": "% de empates", "gols_pro_jogo": "gols pró",
            "gols_contra_jogo": "gols sofridos"}

# Punições do STJD que mudam a tabela e não aparecem nos placares (ver regressao.py)
PUNICOES = {(2013, "Portuguesa"): -4, (2013, "Flamengo"): -4}

AZUL, VERMELHO, CINZA = "#2a6fb0", "#c8402f", "#8a8f98"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def linhas_por_time(df):
    m = df.assign(time=df["mandante"], gf=df["mandante_Placar"], ga=df["visitante_Placar"])
    v = df.assign(time=df["visitante"], gf=df["visitante_Placar"], ga=df["mandante_Placar"])
    x = pd.concat([m, v])[["temporada", "time", "gf", "ga"]].copy()
    x["pts"] = np.where(x.gf > x.ga, 3, np.where(x.gf == x.ga, 1, 0))
    x["v"] = (x.gf > x.ga).astype(int)
    x["e"] = (x.gf == x.ga).astype(int)
    return x


def tabela(x, punicoes=None):
    t = x.groupby(["temporada", "time"], as_index=False).agg(
        jogos=("pts", "size"), pontos=("pts", "sum"), vitorias=("v", "sum"),
        empates=("e", "sum"), gols_pro=("gf", "sum"), gols_contra=("ga", "sum"))
    if punicoes:
        t["pontos"] += [punicoes.get(k, 0) for k in zip(t.temporada, t.time)]
    t["saldo"] = t.gols_pro - t.gols_contra
    t = t.sort_values(["temporada", "pontos", "vitorias", "saldo", "gols_pro"],
                      ascending=[True, False, False, False, False])
    t["posicao"] = t.groupby("temporada").cumcount() + 1
    t["aproveitamento"] = t.pontos / (3 * t.jogos) * 100
    t["pct_vitorias"] = t.vitorias / t.jogos * 100
    t["pct_empates"] = t.empates / t.jogos * 100
    t["gols_pro_jogo"] = t.gols_pro / t.jogos
    t["gols_contra_jogo"] = t.gols_contra / t.jogos
    return t


# 1) Tabelas finais 2006-2025
hist = pd.read_csv(PASTA / "brasileirao_limpo.csv", dtype={"ID": "string"})
hist = hist[(hist.temporada >= 2006) & (hist.temporada <= 2025)]
hist = hist[~hist["observacao"].fillna("").str.startswith("W.O.")]   # W.O. duplo de 2016
final = tabela(linhas_por_time(hist), PUNICOES)
corte = final[final.posicao == 20 - N_REBAIXADOS + 1].set_index("temporada")["pontos"]   # 17º
ultimo_salvo = final[final.posicao == 20 - N_REBAIXADOS].set_index("temporada")["pontos"]  # 16º

d = final[final.posicao > 12].copy()
d["rebaixado"] = (d.posicao > 20 - N_REBAIXADOS).astype(int)
d["grupo"] = np.where(d.rebaixado == 1, "rebaixado", "escapou por pouco")
d["pontos_17o_do_ano"] = d.temporada.map(corte)
d["pontos_16o_do_ano"] = d.temporada.map(ultimo_salvo)
print(f"Base: {len(d)} campanhas (13º ao 20º, {d.temporada.min()}-{d.temporada.max()})")
print(f"Linha de corte (pontos do 17º): média {corte.mean():.1f}, "
      f"de {corte.min()} ({corte.idxmin()}) a {corte.max()} ({corte.idxmax()})")

# 2) Método 1: em quantas das outras temporadas essa pontuação mudaria o destino
def pct_outro_destino(temporada, pontos, rebaixado):
    outras = corte.drop(temporada)
    salvaria = pontos > outras
    return float((salvaria if rebaixado else ~salvaria).mean())


d["pct_outro_destino"] = [pct_outro_destino(t, p, r)
                          for t, p, r in zip(d.temporada, d.pontos, d.rebaixado)]
d["anomalia_pontos"] = np.where(d.pct_outro_destino >= LIMITE_OUTRO_DESTINO, "sim", "nao")
d["tipo"] = np.select(
    [(d.anomalia_pontos == "sim") & (d.rebaixado == 1),
     (d.anomalia_pontos == "sim") & (d.rebaixado == 0)],
    ["caiu com muitos pontos", "escapou com poucos pontos"], default="")

COLS = ["temporada", "time", "posicao", "pontos", "saldo", "pontos_17o_do_ano",
        "pct_outro_destino"]
fmt = lambda df: df.assign(pct_outro_destino=(df.pct_outro_destino * 100).round(0).astype(int)
                           .astype(str) + "%").to_string(index=False)
caiu = d[d.tipo == "caiu com muitos pontos"].sort_values(["pct_outro_destino", "pontos"],
                                                          ascending=False)
escapou = d[d.tipo == "escapou com poucos pontos"].sort_values(["pct_outro_destino", "pontos"],
                                                                ascending=[False, True])
print(f"\nMétodo 1 - CAIU COM MUITOS PONTOS (pontuação que salvaria em >= "
      f"{LIMITE_OUTRO_DESTINO:.0%} das outras temporadas):")
print(fmt(caiu[COLS]))
print(f"\nMétodo 1 - ESCAPOU COM POUCOS PONTOS (pontuação que derrubaria em >= "
      f"{LIMITE_OUTRO_DESTINO:.0%} das outras temporadas):")
print(fmt(escapou[COLS]))

# 3) Método 2: Isolation Forest com o perfil da campanha
escala = StandardScaler().fit(d[FEATURES])
Z = escala.transform(d[FEATURES])
iso = IsolationForest(n_estimators=500, contamination=CONTAMINACAO, random_state=0).fit(Z)
d["score_if"] = -iso.score_samples(Z)       # maior = mais isolada
d["anomalia_if"] = np.where(iso.predict(Z) == -1, "sim", "nao")
mais = np.abs(Z).argmax(axis=1)
d["motivo_if"] = [f"{NOME_VAR[FEATURES[j]]} muito {'alto' if Z[i, j] > 0 else 'baixo'} "
                  f"(z = {Z[i, j]:+.1f})" for i, j in enumerate(mais)]
d.loc[d.anomalia_if == "nao", "motivo_if"] = ""

print(f"\nMétodo 2 - ISOLATION FOREST ({(d.anomalia_if == 'sim').sum()} campanhas com perfil "
      f"mais atípico):")
print(d[d.anomalia_if == "sim"].sort_values("score_if", ascending=False)[
    ["temporada", "time", "posicao", "pontos", "gols_pro", "gols_contra", "empates",
     "score_if", "motivo_if"]].round(3).to_string(index=False))

dois = d[(d.anomalia_pontos == "sim") & (d.anomalia_if == "sim")]
print(f"\nMarcadas pelos dois métodos: {len(dois)}"
      + ("" if dois.empty else " -> " + ", ".join(f"{t} {a}" for t, a in zip(dois.time, dois.temporada))))

saida = d[["temporada", "time", "posicao", "grupo", "pontos", "jogos", "vitorias", "empates",
           "gols_pro", "gols_contra", "saldo", *FEATURES, "pontos_17o_do_ano",
           "pontos_16o_do_ano", "pct_outro_destino", "anomalia_pontos", "tipo", "score_if",
           "anomalia_if", "motivo_if"]].copy()
saida["pct_outro_destino"] = (saida.pct_outro_destino * 100).round(1)
saida.round(3).to_csv(SAIDA / "anomalias_campanhas.csv", index=False, encoding="utf-8-sig")

# 4) Gráfico: pontos dos 8 últimos por temporada
fig, ax = plt.subplots(figsize=(13, 6.5))
for flag, cor, nome in ((0, CINZA, "escapou (13º-16º)"), (1, VERMELHO, "rebaixado (17º-20º)")):
    s = d[d.rebaixado == flag]
    ax.scatter(s.temporada, s.pontos, c=cor, s=26, alpha=.75, label=nome, zorder=3)
s = d[d.anomalia_if == "sim"]
ax.scatter(s.temporada, s.pontos, marker="s", s=90, facecolors="none", edgecolors="black",
           lw=.9, label="perfil atípico (Isolation Forest)", zorder=4)
ax.axhline(ultimo_salvo.mean(), color=AZUL, ls="--", lw=1,
           label=f"média do 16º colocado ({ultimo_salvo.mean():.1f} pts)")
ax.axhline(corte.mean(), color=VERMELHO, ls=":", lw=1,
           label=f"média do 17º colocado ({corte.mean():.1f} pts)")
# nomes no mesmo ponto viram um rótulo só. Rebaixados: rótulo acima, em 3 alturas alternadas;
# escaparam: rótulo à direita, no espaço entre uma temporada e outra
rotulos = (pd.concat([caiu, escapou]).groupby(["rebaixado", "temporada", "pontos"])["time"]
           .agg(" e ".join).reset_index().sort_values("temporada"))
for cima, g in rotulos.groupby("rebaixado"):
    for i, r in enumerate(g.itertuples()):
        pos, ha = ((0, 10 + 14 * (i % 3)), "center") if cima else ((8, -3), "left")
        ax.annotate(f"{r.time} ({r.pontos})", (r.temporada, r.pontos), fontsize=8,
                    xytext=pos, textcoords="offset points", ha=ha,
                    color=VERMELHO if cima else "black", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="gray", lw=.5) if cima else None)
ax.set_xticks(sorted(d.temporada.unique()))
ax.tick_params(axis="x", rotation=60)
ax.set(xlabel="Temporada", ylabel="Pontos no fim do campeonato",
       title="Anomalias no rebaixamento: quem caiu com muitos pontos e quem escapou com poucos")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.32), fontsize=9)
fig.tight_layout()
fig.savefig(SAIDA / "anomalias_rebaixamento.png", dpi=130, bbox_inches="tight")
print("\nArquivos salvos em:", SAIDA.resolve())
