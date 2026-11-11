"""Análise de agrupamento (K-Means) dos times do Brasileirão por perfil de desempenho.

Cada ponto é um time numa temporada (2006-2025, campanha completa). As variáveis são medidas
por jogo, então dá para comparar campanhas completas com a campanha parcial de 2026.
O K-Means NÃO recebe a informação de quem caiu. Depois do agrupamento comparamos cada grupo
com o rebaixamento real, para ver se os perfis encontrados fazem sentido.

Passos:
  1. monta a tabela final de cada temporada e calcula as variáveis por jogo
  2. padroniza as variáveis (média 0, desvio 1)
  3. testa k = 2..8 (inércia/cotovelo e silhueta) e usa K grupos
  4. descreve cada grupo (médias, % de rebaixados, posição média)
  5. coloca os times de 2026 no grupo mais próximo

Arquivos de entrada (na mesma pasta do script):
  brasileirao_limpo.csv        histórico limpo (gerado por brasileirao_pipeline.py)
  brasileirao_2026_jogos.csv   jogos de 2026 (gerado por coletar_2026.py), opcional
Saídas:
  agrupamento_escolha_k.csv     inércia e silhueta para cada k
  agrupamento_perfis.csv        perfil médio de cada grupo
  agrupamento_times.csv         cada time-temporada com seu grupo
  agrupamento_2026.csv          grupo de cada time em 2026
  agrupamento_escolha_k.png     gráficos do cotovelo e da silhueta
  agrupamento_pca.png           grupos projetados em 2 dimensões (PCA)

Uso:  python agrupamento.py [pasta_de_saida] [k]
Requer: pandas, scikit-learn, matplotlib
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

N_REBAIXADOS = 4
PASTA = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4   # 4 perfis: fácil de interpretar

FEATURES = ["aproveitamento", "pct_vitorias", "pct_empates", "gols_pro_jogo",
            "gols_contra_jogo", "saldo_por_jogo"]


def linhas_por_time(df):
    m = df.assign(time=df["mandante"], gf=df["mandante_Placar"], ga=df["visitante_Placar"])
    v = df.assign(time=df["visitante"], gf=df["visitante_Placar"], ga=df["mandante_Placar"])
    x = pd.concat([m, v])[["temporada", "time", "gf", "ga"]].copy()
    x["pts"] = np.where(x.gf > x.ga, 3, np.where(x.gf == x.ga, 1, 0))
    x["v"] = (x.gf > x.ga).astype(int)
    x["e"] = (x.gf == x.ga).astype(int)
    return x


def tabela(x):
    t = x.groupby(["temporada", "time"], as_index=False).agg(
        jogos=("pts", "size"), pontos=("pts", "sum"), vitorias=("v", "sum"),
        empates=("e", "sum"), gols_pro=("gf", "sum"), gols_contra=("ga", "sum"))
    t["saldo"] = t.gols_pro - t.gols_contra
    t = t.sort_values(["temporada", "pontos", "vitorias", "saldo", "gols_pro"],
                      ascending=[True, False, False, False, False])
    t["posicao"] = t.groupby("temporada").cumcount() + 1
    t["aproveitamento"] = t.pontos / (3 * t.jogos) * 100
    t["pct_vitorias"] = t.vitorias / t.jogos * 100
    t["pct_empates"] = t.empates / t.jogos * 100
    t["gols_pro_jogo"] = t.gols_pro / t.jogos
    t["gols_contra_jogo"] = t.gols_contra / t.jogos
    t["saldo_por_jogo"] = t.saldo / t.jogos
    return t


# 1) Histórico 2006-2025 (formato com 20 clubes e 4 rebaixados)
hist = pd.read_csv(PASTA / "brasileirao_limpo.csv", dtype={"ID": "string"})
hist = hist[(hist.temporada >= 2006) & (hist.temporada <= 2025)]
# W.O. duplo de 2016 não tem placar real: fica fora do cálculo de perfil
hist = hist[~hist["observacao"].fillna("").str.startswith("W.O.")]
dados = tabela(linhas_por_time(hist))
dados["rebaixado"] = (dados.posicao > 20 - N_REBAIXADOS).astype(int)
print(f"Base: {len(dados)} times-temporada ({dados.temporada.min()}-{dados.temporada.max()})")

# 2) Padronização
escala = StandardScaler().fit(dados[FEATURES])
X = escala.transform(dados[FEATURES])

# 3) Escolha de k
escolha = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
    escolha.append(dict(k=k, inercia=round(km.inertia_, 1),
                        silhueta=round(silhouette_score(X, km.labels_), 3)))
escolha = pd.DataFrame(escolha)
print("\nEscolha do número de grupos:")
print(escolha.to_string(index=False))
escolha.to_csv(SAIDA / "agrupamento_escolha_k.csv", index=False, encoding="utf-8-sig")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(escolha.k, escolha.inercia, "o-"); a1.set_title("Cotovelo (inércia)"); a1.set_xlabel("k")
a2.plot(escolha.k, escolha.silhueta, "o-"); a2.set_title("Silhueta"); a2.set_xlabel("k")
for a in (a1, a2):
    a.axvline(K, color="gray", ls="--")
fig.tight_layout()
fig.savefig(SAIDA / "agrupamento_escolha_k.png", dpi=130)

# 4) Modelo final com K grupos; grupos numerados do pior (1) para o melhor desempenho
km = KMeans(n_clusters=K, n_init=50, random_state=0).fit(X)
ordem = np.argsort(escala.inverse_transform(km.cluster_centers_)[:, 0])  # por aproveitamento
renum = {int(c): i + 1 for i, c in enumerate(ordem)}
dados["grupo"] = [renum[int(c)] for c in km.labels_]

# Nomes dados após olhar os perfis (k=4): os grupos 2 e 3 têm aproveitamento parecido e se
# diferenciam pelo estilo (muitos empates x mais vitórias e mais gols sofridos)
NOMES_4 = {1: "Luta contra o rebaixamento", 2: "Meio de tabela - muitos empates",
           3: "Meio de tabela - jogos abertos", 4: "Briga pelo título / G4"}
nome_grupo = (lambda g: NOMES_4[g]) if K == 4 else (lambda g: f"Grupo {g}")
dados["perfil"] = dados.grupo.map(nome_grupo)

perfis = dados.groupby(["grupo", "perfil"]).agg(
    times=("time", "size"), **{f: (f, "mean") for f in FEATURES},
    posicao_media=("posicao", "mean"), pontos_medios=("pontos", "mean"),
    pct_rebaixados=("rebaixado", "mean")).reset_index()
perfis["pct_rebaixados"] *= 100
perfis["pct_do_total_de_rebaixados"] = (
    dados.groupby("grupo").rebaixado.sum().values / dados.rebaixado.sum() * 100)
perfis = perfis.round(2)
print(f"\nPerfil dos {K} grupos:")
print(perfis.to_string(index=False))
perfis.to_csv(SAIDA / "agrupamento_perfis.csv", index=False, encoding="utf-8-sig")
dados[["temporada", "time", "posicao", "pontos", *FEATURES, "rebaixado", "grupo", "perfil"]] \
    .round(2).to_csv(SAIDA / "agrupamento_times.csv", index=False, encoding="utf-8-sig")

# 5) Visualização em 2D (PCA)
pca = PCA(n_components=2).fit(X)
P = pca.transform(X)
fig, ax = plt.subplots(figsize=(9, 7))
for g in sorted(dados.grupo.unique()):
    s = (dados.grupo == g).values
    ax.scatter(P[s, 0], P[s, 1], s=18, alpha=0.6, label=nome_grupo(g))
r = dados.rebaixado.values == 1
ax.scatter(P[r, 0], P[r, 1], s=40, facecolors="none", edgecolors="black", lw=0.8,
           label="rebaixado (real)")

# 6) 2026: coloca cada time no grupo mais próximo
arq26 = PASTA / "brasileirao_2026_jogos.csv"
if arq26.exists():
    t26 = tabela(linhas_por_time(pd.read_csv(arq26)))
    X26 = escala.transform(t26[FEATURES])
    t26["grupo"] = [renum[int(c)] for c in km.predict(X26)]
    t26["perfil"] = t26.grupo.map(nome_grupo)
    t26["distancia_centro"] = np.linalg.norm(X26 - km.cluster_centers_[km.predict(X26)], axis=1)
    taxa = perfis.set_index("grupo")["pct_rebaixados"]
    t26["pct_rebaixados_historico_do_grupo"] = t26.grupo.map(taxa)
    s26 = t26.sort_values(["grupo", "aproveitamento"])[
        ["grupo", "perfil", "posicao", "time", "jogos", "pontos", *FEATURES,
         "distancia_centro", "pct_rebaixados_historico_do_grupo"]].round(2)
    print("\nTimes de 2026 por grupo (do pior para o melhor perfil):")
    print(s26[["grupo", "perfil", "posicao", "time", "pontos", "aproveitamento",
               "pct_rebaixados_historico_do_grupo"]].to_string(index=False))
    s26.to_csv(SAIDA / "agrupamento_2026.csv", index=False, encoding="utf-8-sig")
    P26 = pca.transform(X26)
    ax.scatter(P26[:, 0], P26[:, 1], marker="*", s=160, c="red", edgecolors="black",
               label="2026 (atual)")
    for (x, y), nome in zip(P26, t26.time):
        ax.annotate(nome, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

ev = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"Componente 1 ({ev[0]:.0f}% da variância)")
ax.set_ylabel(f"Componente 2 ({ev[1]:.0f}% da variância)")
ax.set_title(f"K-Means com {K} grupos – times-temporada 2006-2025")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(SAIDA / "agrupamento_pca.png", dpi=130)
print("\nArquivos salvos em:", SAIDA.resolve())
