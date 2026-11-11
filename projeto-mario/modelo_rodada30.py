"""Previsão de rebaixamento a partir da situação do time na rodada 30.

Modelos: árvore de decisão rasa (explicativa), Random Forest e regressão logística (referência).
Validação: deixa uma temporada inteira de fora por vez (leave-one-season-out).
Usa 2006-2025 (20 clubes, 38 rodadas, 4 rebaixados), para manter o formato comparável.

Uso:  python modelo_rodada30.py [pasta_de_saida]
Requer: pandas, scikit-learn, matplotlib
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

RODADA = 30
N_REBAIXADOS = 4
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)
ENTRADA = Path(__file__).resolve().parent / "brasileirao_limpo.csv"

jogos = pd.read_csv(ENTRADA, dtype={"ID": "string"})
jogos = jogos[(jogos["temporada"] >= 2006)].copy()


def linhas_por_time(df):
    """Uma linha por time por jogo, com pontos, gols e rodada."""
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
    # desempate: pontos, vitórias, saldo, gols pró
    t = t.sort_values(["temporada", "pontos", "vitorias", "saldo", "gols_pro"],
                      ascending=[True, False, False, False, False])
    t["posicao"] = t.groupby("temporada").cumcount() + 1
    return t


todos = linhas_por_time(jogos)
final = tabela(todos)
final["rebaixado"] = (final["posicao"] > 20 - N_REBAIXADOS).astype(int)

parcial = tabela(todos[todos["rodata"] <= RODADA])
# pontos do 16º colocado (primeiro time fora da zona) na rodada 30
p16 = parcial[parcial.posicao == 20 - N_REBAIXADOS].set_index("temporada")["pontos"]
parcial["pontos_sobre_zona"] = parcial["pontos"] - parcial["temporada"].map(p16)
parcial["aproveitamento"] = parcial["pontos"] / (3 * parcial["jogos"]) * 100
parcial["saldo_por_jogo"] = parcial["saldo"] / parcial["jogos"]
parcial["gols_pro_jogo"] = parcial["gols_pro"] / parcial["jogos"]
parcial["gols_contra_jogo"] = parcial["gols_contra"] / parcial["jogos"]
parcial["pct_vitorias"] = parcial["vitorias"] / parcial["jogos"] * 100
parcial["pct_derrotas"] = parcial["derrotas"] / parcial["jogos"] * 100

FEATURES = ["aproveitamento", "posicao", "pontos_sobre_zona", "saldo_por_jogo",
            "gols_pro_jogo", "gols_contra_jogo", "pct_vitorias", "pct_derrotas"]
dados = parcial.merge(final[["temporada", "time", "rebaixado", "posicao", "pontos"]],
                      on=["temporada", "time"], suffixes=("", "_final"))
print(f"Base: {len(dados)} times-temporada, {dados.rebaixado.sum()} rebaixados "
      f"({dados.rebaixado.mean():.0%}), temporadas {dados.temporada.min()}-{dados.temporada.max()}")

MODELOS = {
    "Arvore (prof. 3)": lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                                       class_weight="balanced", random_state=0),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                    class_weight="balanced", random_state=0),
    "Regressao logistica": lambda: make_pipeline(StandardScaler(),
                                                 LogisticRegression(class_weight="balanced",
                                                                    max_iter=1000)),
}

# Validação: uma temporada inteira de fora por vez
prev = dados[["temporada", "time", "rebaixado"]].copy()
for nome, fabrica in MODELOS.items():
    p = np.zeros(len(dados))
    for t in sorted(dados.temporada.unique()):
        teste = (dados.temporada == t).values
        m = fabrica().fit(dados.loc[~teste, FEATURES], dados.loc[~teste, "rebaixado"])
        p[teste] = m.predict_proba(dados.loc[teste, FEATURES])[:, 1]
    prev[nome] = p

resumo = []
for nome in MODELOS:
    acertos = []
    for t, g in prev.groupby("temporada"):
        top4 = g.nlargest(N_REBAIXADOS, nome)
        acertos.append(int(top4["rebaixado"].sum()))
    resumo.append(dict(modelo=nome,
                       auc=round(roc_auc_score(prev.rebaixado, prev[nome]), 3),
                       brier=round(brier_score_loss(prev.rebaixado, prev[nome]), 3),
                       acertos_medios_dos_4=round(float(np.mean(acertos)), 2),
                       temporadas_com_4_de_4=int(sum(a == 4 for a in acertos)),
                       temporadas_com_3_ou_mais=int(sum(a >= 3 for a in acertos))))
resumo = pd.DataFrame(resumo)
print("\nValidação (leave-one-season-out, rodada 30):")
print(resumo.to_string(index=False))
resumo.to_csv(SAIDA / "modelo_validacao.csv", index=False, encoding="utf-8-sig")
prev.round(4).to_csv(SAIDA / "modelo_previsoes_validacao.csv", index=False, encoding="utf-8-sig")

# Modelos finais com todos os dados: regras da árvore e importâncias
arvore = MODELOS["Arvore (prof. 3)"]().fit(dados[FEATURES], dados.rebaixado)
regras = export_text(arvore, feature_names=FEATURES)
(SAIDA / "modelo_regras_arvore.txt").write_text(regras, encoding="utf-8")
print("\nRegras da árvore:\n" + regras)
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(arvore, feature_names=FEATURES, class_names=["fica", "cai"], filled=True,
          rounded=True, fontsize=9, ax=ax)
fig.savefig(SAIDA / "modelo_arvore.png", dpi=130, bbox_inches="tight")

rf = MODELOS["Random Forest"]().fit(dados[FEATURES], dados.rebaixado)
imp = pd.DataFrame({"variavel": FEATURES, "importancia_rf": rf.feature_importances_.round(3),
                    "importancia_arvore": arvore.feature_importances_.round(3)}
                   ).sort_values("importancia_rf", ascending=False)
print("\nImportância das variáveis:\n" + imp.to_string(index=False))
imp.to_csv(SAIDA / "modelo_importancias.csv", index=False, encoding="utf-8-sig")
print("\nArquivos salvos em:", SAIDA.resolve())
