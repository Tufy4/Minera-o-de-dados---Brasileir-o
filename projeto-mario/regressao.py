"""Regressão: prevê com quantos pontos cada time vai terminar o campeonato, a partir da situação
dele numa rodada de corte (padrão: rodada 19, fim do 1º turno).

Foco no rebaixamento: em cada temporada, os 4 times com menos pontos previstos são os
"rebaixados previstos", e a previsão é comparada com a linha histórica de rebaixamento
(pontos médios do 16º colocado, o último que escapa).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_REBAIXADOS = 4
JOGOS_NO_ANO = 38
PASTA = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)
RODADA = int(sys.argv[2]) if len(sys.argv) > 2 else 19
RODADAS_CURVA = [5, 10, 15, 19, 25, 30, 34]

# Punições do STJD que mudam a tabela e não aparecem nos placares. Em 2013, Portuguesa e Flamengo
# perderam 4 pontos cada por escalação irregular na última rodada: na tabela oficial a Portuguesa
# caiu (44 pontos) e o Fluminense (46) ficou. Sem isso, a tabela calculada pelos placares
# rebaixaria o Fluminense.
PUNICOES = {(2013, "Portuguesa"): -4, (2013, "Flamengo"): -4}

# Sem "pontos", "saldo" e "% de empates": numa rodada fixa eles são combinação exata de outras
# variáveis (ex.: saldo = gols pró - gols contra), o que deixa os coeficientes da regressão
# linear sem sentido.
FEATURES = ["aproveitamento", "pct_vitorias", "gols_pro_jogo", "gols_contra_jogo",
            "posicao", "pontos_sobre_zona"]
NOMES = ["Projecao simples", "Regressao linear", "Random Forest"]
MODELOS = {
    "Regressao linear": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                   random_state=0, n_jobs=-1),
}
PRINCIPAL = "Regressao linear"

AZUL, VERMELHO, CINZA, VERDE = "#2a6fb0", "#c8402f", "#8a8f98", "#3a9a5b"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def linhas_por_time(df):
    m = df.assign(time=df["mandante"], gf=df["mandante_Placar"], ga=df["visitante_Placar"])
    v = df.assign(time=df["visitante"], gf=df["visitante_Placar"], ga=df["mandante_Placar"])
    x = pd.concat([m, v])[["temporada", "rodata", "time", "gf", "ga"]].copy()
    x["pts"] = np.where(x.gf > x.ga, 3, np.where(x.gf == x.ga, 1, 0))
    x["v"] = (x.gf > x.ga).astype(int)
    return x


def tabela(x, punicoes=None):
    t = x.groupby(["temporada", "time"], as_index=False).agg(
        jogos=("pts", "size"), pontos=("pts", "sum"), vitorias=("v", "sum"),
        gols_pro=("gf", "sum"), gols_contra=("ga", "sum"))
    if punicoes:
        t["pontos"] += [punicoes.get(k, 0) for k in zip(t.temporada, t.time)]
    t["saldo"] = t.gols_pro - t.gols_contra
    t = t.sort_values(["temporada", "pontos", "vitorias", "saldo", "gols_pro"],
                      ascending=[True, False, False, False, False])
    t["posicao"] = t.groupby("temporada").cumcount() + 1
    return t


def com_features(t):
    p16 = t[t.posicao == 20 - N_REBAIXADOS].set_index("temporada")["pontos"]
    t = t.copy()
    t["pontos_sobre_zona"] = t["pontos"] - t["temporada"].map(p16)
    t["aproveitamento"] = t["pontos"] / (3 * t["jogos"]) * 100
    t["pct_vitorias"] = t["vitorias"] / t["jogos"] * 100
    t["gols_pro_jogo"] = t["gols_pro"] / t["jogos"]
    t["gols_contra_jogo"] = t["gols_contra"] / t["jogos"]
    return t


def situacao_na_rodada(x, final, rodada):
    """Situação de cada time na rodada de corte + pontos e posição no fim do campeonato."""
    parcial = com_features(tabela(x[x["rodata"] <= rodada]))
    fim = final[["temporada", "time", "pontos", "posicao", "rebaixado"]].rename(
        columns={"pontos": "pontos_final", "posicao": "posicao_final"})
    return parcial.merge(fim, on=["temporada", "time"])


def validar(dados):
    """Previsão de cada time-temporada com o modelo treinado nas outras temporadas."""
    prev = dados[["temporada", "time", "posicao", "pontos", "pontos_final", "posicao_final",
                  "rebaixado"]].copy()
    prev["Projecao simples"] = dados["pontos"] / dados["jogos"] * JOGOS_NO_ANO
    for nome, fabrica in MODELOS.items():
        p = np.zeros(len(dados))
        for t in sorted(dados.temporada.unique()):
            teste = (dados.temporada == t).values
            m = fabrica().fit(dados.loc[~teste, FEATURES], dados.loc[~teste, "pontos_final"])
            p[teste] = m.predict(dados.loc[teste, FEATURES])
        prev[nome] = p
    return prev


def metricas(prev, nome):
    y, p = prev["pontos_final"], prev[nome]
    baixo = prev["posicao_final"] > 20 - 8
    acertos = [int(g.nsmallest(N_REBAIXADOS, nome)["rebaixado"].sum())
               for _, g in prev.groupby("temporada")]
    return dict(modelo=nome,
                mae=round(mean_absolute_error(y, p), 2),
                rmse=round(mean_squared_error(y, p) ** 0.5, 2),
                r2=round(r2_score(y, p), 3),
                mae_8_ultimos=round(mean_absolute_error(y[baixo], p[baixo]), 2),
                acertos_medios_dos_4=round(float(np.mean(acertos)), 2),
                temporadas_com_4_de_4=sum(a == 4 for a in acertos),
                temporadas_com_3_ou_mais=sum(a >= 3 for a in acertos))


# 1) Histórico 2006-2025 (20 clubes, 38 rodadas, 4 rebaixados)
hist = pd.read_csv(PASTA / "brasileirao_limpo.csv", dtype={"ID": "string"})
hist = hist[(hist.temporada >= 2006) & (hist.temporada <= 2025)]
# W.O. duplo de 2016: nenhum dos dois pontuou, então o jogo fica fora (os pontos ficam certos)
hist = hist[~hist["observacao"].fillna("").str.startswith("W.O.")]
x_hist = linhas_por_time(hist)
final = tabela(x_hist, PUNICOES)
final["rebaixado"] = (final.posicao > 20 - N_REBAIXADOS).astype(int)

linha16 = final.loc[final.posicao == 20 - N_REBAIXADOS, "pontos"].mean()
linha17 = final.loc[final.posicao == 20 - N_REBAIXADOS + 1, "pontos"].mean()
print(f"Linha histórica: o 16º (último a escapar) fez em média {linha16:.1f} pontos; "
      f"o 17º (primeiro rebaixado), {linha17:.1f}.")

# 2) Validação na rodada de corte
dados = situacao_na_rodada(x_hist, final, RODADA)
print(f"Base: {len(dados)} times-temporada, situação na rodada {RODADA}")
prev = validar(dados)
resumo = pd.DataFrame([metricas(prev, n) for n in NOMES])
print(f"\nValidação (uma temporada de fora por vez), rodada {RODADA}:")
print(resumo.to_string(index=False))
resumo.to_csv(SAIDA / "regressao_validacao.csv", index=False, encoding="utf-8-sig")

prev["erro_" + PRINCIPAL] = prev[PRINCIPAL] - prev["pontos_final"]
prev["abaixo_da_linha_previsto"] = np.where(prev[PRINCIPAL] < linha16, "sim", "nao")
prev.round(2).to_csv(SAIDA / "regressao_previsoes.csv", index=False, encoding="utf-8-sig")

# Maiores erros entre os rebaixados: quem caiu "sem o modelo esperar"
caiu = prev[prev.rebaixado == 1].sort_values("erro_" + PRINCIPAL, ascending=False)
print(f"\nRebaixados com previsão mais otimista na rodada {RODADA} ({PRINCIPAL}):")
print(caiu[["temporada", "time", "posicao", "pontos", PRINCIPAL, "pontos_final",
            "posicao_final"]].head(5).round(1).to_string(index=False))

# 3) Variáveis: coeficientes (variáveis padronizadas) e importâncias
lin = MODELOS["Regressao linear"]().fit(dados[FEATURES], dados["pontos_final"])
rf = MODELOS["Random Forest"]().fit(dados[FEATURES], dados["pontos_final"])
var = pd.DataFrame({"variavel": FEATURES,
                    "coef_linear_padronizado": lin[-1].coef_.round(2),
                    "importancia_rf": rf.feature_importances_.round(3)}
                   ).sort_values("importancia_rf", ascending=False)
print(f"\nVariáveis (intercepto da regressão linear: {lin[-1].intercept_:.1f} pontos):")
print(var.to_string(index=False))
var.to_csv(SAIDA / "regressao_variaveis.csv", index=False, encoding="utf-8-sig")

# 4) Erro conforme a rodada de corte: quanto antes, maior a incerteza
curva = []
for r in RODADAS_CURVA:
    pr = prev if r == RODADA else validar(situacao_na_rodada(x_hist, final, r))
    curva += [dict(rodada=r, **metricas(pr, n)) for n in NOMES]
curva = pd.DataFrame(curva)
print("\nMAE (pontos) por rodada de corte:")
print(curva.pivot(index="rodada", columns="modelo", values="mae")[NOMES].to_string())
curva.to_csv(SAIDA / "regressao_por_rodada.csv", index=False, encoding="utf-8-sig")

# 5) Gráficos
fig, ax = plt.subplots(figsize=(7.5, 7))
for flag, cor, nome in ((0, CINZA, "não caiu"), (1, VERMELHO, "rebaixado")):
    s = prev[prev.rebaixado == flag]
    ax.scatter(s[PRINCIPAL], s["pontos_final"], s=16, alpha=.6, c=cor, label=nome)
lim = [prev[[PRINCIPAL, "pontos_final"]].min().min() - 3,
       prev[[PRINCIPAL, "pontos_final"]].max().max() + 3]
ax.plot(lim, lim, color="black", lw=.8, label="previsão perfeita")
ax.axhline(linha16, color=AZUL, ls="--", lw=.9, label=f"linha histórica ({linha16:.1f} pts)")
ax.axvline(linha16, color=AZUL, ls="--", lw=.9)
ax.set(xlim=lim, ylim=lim, xlabel=f"Pontos previstos na rodada {RODADA}",
       ylabel="Pontos no fim do campeonato",
       title=f"{PRINCIPAL}: previsto x real (validação 2006-2025)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(SAIDA / "regressao_previsto_vs_real.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 4.8))
for nome, cor in zip(NOMES, (CINZA, AZUL, VERDE)):
    c = curva[curva.modelo == nome]
    ax.plot(c.rodada, c.mae, "o-", color=cor, label=nome)
ax.set(xlabel="Rodada de corte", ylabel="Erro médio absoluto (pontos)",
       title="Erro da previsão dos pontos finais conforme a rodada")
ax.set_xticks(RODADAS_CURVA)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(SAIDA / "regressao_erro_por_rodada.png", dpi=130)

# 6) 2026: treina na rodada equivalente aos jogos já disputados e prevê os pontos finais
arq26 = PASTA / "brasileirao_2026_jogos.csv"
if arq26.exists():
    atual = com_features(tabela(linhas_por_time(pd.read_csv(arq26))))
    r26 = int(round(atual["jogos"].median()))
    d26 = situacao_na_rodada(x_hist, final, r26)
    mae26 = metricas(validar(d26), PRINCIPAL)["mae"]
    atual["Projecao simples"] = atual["pontos"] / atual["jogos"] * JOGOS_NO_ANO
    for nome, fabrica in MODELOS.items():
        atual[nome] = fabrica().fit(d26[FEATURES], d26["pontos_final"]).predict(atual[FEATURES])
    atual["pontos_previstos"] = atual[PRINCIPAL]
    atual["linha_historica"] = round(linha16, 1)
    atual["abaixo_da_linha"] = np.where(atual["pontos_previstos"] < linha16, "sim", "nao")
    atual = atual.sort_values("pontos_previstos")
    atual["rebaixado_previsto"] = ["sim"] * N_REBAIXADOS + ["nao"] * (len(atual) - N_REBAIXADOS)
    s26 = atual[["posicao", "time", "jogos", "pontos", *NOMES, "pontos_previstos",
                 "linha_historica", "abaixo_da_linha", "rebaixado_previsto"]].round(1)
    s26.to_csv(SAIDA / "regressao_2026.csv", index=False, encoding="utf-8-sig")
    print(f"\n2026 (modelo treinado na rodada {r26}; erro médio esperado: {mae26} pontos):")
    print(s26[["posicao", "time", "jogos", "pontos", "pontos_previstos", "abaixo_da_linha",
               "rebaixado_previsto"]].to_string(index=False))

print("\nArquivos salvos em:", SAIDA.resolve())
