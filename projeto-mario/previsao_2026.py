"""Situação atual do Brasileirão 2026 e chance de rebaixamento de cada time.

Treina árvore de decisão (prof. 3), Random Forest e regressão logística com 2006-2025, olhando a
situação de cada time na rodada RODADA_CORTE (a mais próxima do ponto em que 2026 está), e aplica
os modelos à tabela de 2026 com todos os jogos já disputados.

Arquivos de entrada (na mesma pasta do script):
  brasileirao_limpo.csv        histórico limpo (gerado por brasileirao_pipeline.py)
  brasileirao_2026_jogos.csv   jogos de 2026 (gerado por coletar_2026.py)
Saída:  situacao_atual_2026.csv (ordenado do maior para o menor risco de rebaixamento)

Uso:  python previsao_2026.py [pasta_de_saida]
Requer: pandas, scikit-learn
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RODADA_CORTE = 34      # 2026 está com ~34 jogos por time
N_REBAIXADOS = 4
PASTA = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)

FEATURES = ["aproveitamento", "posicao", "pontos_sobre_zona", "saldo_por_jogo",
            "gols_pro_jogo", "gols_contra_jogo", "pct_vitorias", "pct_derrotas"]


def linhas_por_time(df):
    m = df.assign(time=df["mandante"], gf=df["mandante_Placar"], ga=df["visitante_Placar"])
    v = df.assign(time=df["visitante"], gf=df["visitante_Placar"], ga=df["mandante_Placar"])
    x = pd.concat([m, v])[["temporada", "rodata", "time", "gf", "ga"]].copy()
    x["pts"] = np.where(x.gf > x.ga, 3, np.where(x.gf == x.ga, 1, 0))
    x["v"] = (x.gf > x.ga).astype(int)
    x["e"] = (x.gf == x.ga).astype(int)
    x["d"] = (x.gf < x.ga).astype(int)
    return x


def tabela(x):
    t = x.groupby(["temporada", "time"], as_index=False).agg(
        jogos=("pts", "size"), pontos=("pts", "sum"), vitorias=("v", "sum"),
        empates=("e", "sum"), derrotas=("d", "sum"), gols_pro=("gf", "sum"),
        gols_contra=("ga", "sum"))
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
    t["saldo_por_jogo"] = t["saldo"] / t["jogos"]
    t["gols_pro_jogo"] = t["gols_pro"] / t["jogos"]
    t["gols_contra_jogo"] = t["gols_contra"] / t["jogos"]
    t["pct_vitorias"] = t["vitorias"] / t["jogos"] * 100
    t["pct_derrotas"] = t["derrotas"] / t["jogos"] * 100
    return t


MODELOS = {
    "arvore": lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                             class_weight="balanced", random_state=0),
    "random_forest": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=0),
    "logistica": lambda: make_pipeline(StandardScaler(),
                                       LogisticRegression(class_weight="balanced", max_iter=1000)),
}

# --- Histórico (2006-2025): situação na rodada de corte e se o time caiu no final
hist = pd.read_csv(PASTA / "brasileirao_limpo.csv", dtype={"ID": "string"})
hist = hist[(hist["temporada"] >= 2006) & (hist["temporada"] <= 2025)]
x_hist = linhas_por_time(hist)
final = tabela(x_hist)
final["rebaixado"] = (final["posicao"] > 20 - N_REBAIXADOS).astype(int)
parcial = com_features(tabela(x_hist[x_hist["rodata"] <= RODADA_CORTE]))
dados = parcial.merge(final[["temporada", "time", "rebaixado"]], on=["temporada", "time"])

# Validação rápida (uma temporada de fora por vez) para essa rodada de corte
print(f"Validação com corte na rodada {RODADA_CORTE} (AUC, uma temporada de fora por vez):")
for nome, fab in MODELOS.items():
    p = np.zeros(len(dados))
    for t in sorted(dados.temporada.unique()):
        teste = (dados.temporada == t).values
        m = fab().fit(dados.loc[~teste, FEATURES], dados.loc[~teste, "rebaixado"])
        p[teste] = m.predict_proba(dados.loc[teste, FEATURES])[:, 1]
    print(f"  {nome:14s} AUC = {roc_auc_score(dados.rebaixado, p):.3f}")

# --- 2026: tabela atual e previsão
j26 = pd.read_csv(PASTA / "brasileirao_2026_jogos.csv")
atual = com_features(tabela(linhas_por_time(j26)))
for nome, fab in MODELOS.items():
    m = fab().fit(dados[FEATURES], dados["rebaixado"])
    atual[f"prob_{nome}"] = m.predict_proba(atual[FEATURES])[:, 1]
cols_prob = [f"prob_{n}" for n in MODELOS]
atual["prob_media"] = atual[cols_prob].mean(axis=1)
atual["zona_rebaixamento_hoje"] = np.where(atual.posicao > 20 - N_REBAIXADOS, "sim", "nao")

saida = atual[["posicao", "time", "pontos", "jogos", "vitorias", "empates", "derrotas",
               "gols_pro", "gols_contra", "saldo", "aproveitamento", "pontos_sobre_zona",
               *cols_prob, "prob_media", "zona_rebaixamento_hoje"]].copy()
saida[["aproveitamento", "pontos_sobre_zona"]] = saida[["aproveitamento", "pontos_sobre_zona"]].round(2)
saida[cols_prob + ["prob_media"]] = (saida[cols_prob + ["prob_media"]] * 100).round(1)
# Ordena pelo risco (regressão logística): 1º = maior chance de cair
saida = saida.sort_values(["prob_logistica", "posicao"], ascending=[False, False])
saida.insert(0, "ranking_risco", range(1, len(saida) + 1))
saida.to_csv(SAIDA / "situacao_atual_2026.csv", index=False, encoding="utf-8-sig")

print(f"\nTimes de 2026 do maior para o menor risco de queda ({len(j26)} jogos). Probabilidades em %:")
print(saida[["ranking_risco", "posicao", "time", "pontos", "jogos", "saldo", "aproveitamento",
             "prob_arvore", "prob_random_forest", "prob_logistica", "prob_media"]]
      .to_string(index=False))
print("\nArquivo salvo em:", (SAIDA / "situacao_atual_2026.csv").resolve())
