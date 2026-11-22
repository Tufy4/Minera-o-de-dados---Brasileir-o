"""Regras de associação (Apriori): que características de um time numa rodada de corte (padrão:
rodada 19, fim do 1º turno) aparecem junto com o rebaixamento no fim do campeonato.

Cada campanha (time-temporada, 2006-2025) vira uma "transação" com itens do tipo atributo=valor,
por exemplo {posicao=Z4, saldo=negativo, defesa=fraca, rebaixado=sim}. O Apriori encontra os
conjuntos de itens frequentes e, a partir deles, as regras X -> {rebaixado=sim}.

Atributos (todos medidos na rodada de corte):
  posicao        Z4 (17º-20º), 13a16 (logo acima da zona) ou 1a12
  aproveitamento baixo / medio / alto                (tercis)
  saldo          negativo / zero_ou_positivo
  ataque         fraco / medio / forte               (tercis de gols pró por jogo)
  defesa         fraca / media / forte               (tercis de gols sofridos; fraca = sofre mais)
  empates        poucos / medios / muitos            (tercis do % de empates)
  em_casa        fraco / medio / forte               (tercis do aproveitamento como mandante)
  fora           fraco / medio / forte               (tercis do aproveitamento como visitante)
  rebaixado      sim / nao                           (no fim do campeonato)
Os tercis são calculados sobre todas as campanhas da base; os limites ficam em
regras_discretizacao.csv.

Medidas de cada regra X -> Y:
  suporte     fração das campanhas que têm X e Y
  confianca   entre as campanhas com X, fração que tem Y (para Y = rebaixado: P(caiu | X))
  lift        confiança dividida pela frequência de Y (lift 3 = três vezes mais chance que o normal)
Uma regra é marcada como "redundante" quando não melhora em pelo menos 5 pontos percentuais a
confiança de uma regra mais simples (com parte dos mesmos itens). As listas impressas mostram
só as não redundantes.

O Apriori foi implementado aqui mesmo (função apriori), para não depender de biblioteca extra.

Saídas:
  regras_transacoes.csv      itens de cada campanha
  regras_discretizacao.csv   limites dos tercis
  regras_associacao.csv      regras X -> rebaixado=sim
  regras_associacao.png      suporte x confiança das regras (cor = lift)

Uso:  python regras_associacao.py [pasta_de_saida] [rodada]
Requer: pandas, matplotlib
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_REBAIXADOS = 4
PASTA = Path(__file__).resolve().parent
SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)
RODADA = int(sys.argv[2]) if len(sys.argv) > 2 else 19

MIN_SUPORTE = 0.05      # pelo menos 20 das 400 campanhas (com 3% surgem regras de 12-15 casos)
MIN_CONFIANCA = 0.5
MAX_ITENS = 4           # antecedente com até 3 itens + o consequente
MELHORA_MINIMA = 0.05   # para não ser redundante

# Punições do STJD que mudam a tabela e não aparecem nos placares (ver regressao.py)
PUNICOES = {(2013, "Portuguesa"): -4, (2013, "Flamengo"): -4}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25})


def linhas_por_time(df):
    m = df.assign(time=df["mandante"], gf=df["mandante_Placar"], ga=df["visitante_Placar"],
                  casa=1)
    v = df.assign(time=df["visitante"], gf=df["visitante_Placar"], ga=df["mandante_Placar"],
                  casa=0)
    x = pd.concat([m, v])[["temporada", "rodata", "time", "casa", "gf", "ga"]].copy()
    x["pts"] = np.where(x.gf > x.ga, 3, np.where(x.gf == x.ga, 1, 0))
    x["v"] = (x.gf > x.ga).astype(int)
    x["e"] = (x.gf == x.ga).astype(int)
    x["pts_casa"] = x.pts * x.casa
    x["pts_fora"] = x.pts * (1 - x.casa)
    return x


def tabela(x, punicoes=None):
    t = x.groupby(["temporada", "time"], as_index=False).agg(
        jogos=("pts", "size"), pontos=("pts", "sum"), vitorias=("v", "sum"),
        empates=("e", "sum"), gols_pro=("gf", "sum"), gols_contra=("ga", "sum"),
        jogos_casa=("casa", "sum"), pts_casa=("pts_casa", "sum"), pts_fora=("pts_fora", "sum"))
    if punicoes:
        t["pontos"] += [punicoes.get(k, 0) for k in zip(t.temporada, t.time)]
    t["saldo"] = t.gols_pro - t.gols_contra
    t = t.sort_values(["temporada", "pontos", "vitorias", "saldo", "gols_pro"],
                      ascending=[True, False, False, False, False])
    t["posicao"] = t.groupby("temporada").cumcount() + 1
    return t


def apriori(transacoes, min_suporte, max_itens):
    """Conjuntos de itens frequentes (com seu suporte), nível a nível."""
    n = len(transacoes)

    def suporte(c):
        return sum(1 for t in transacoes if c <= t) / n

    nivel = {frozenset([i]): suporte(frozenset([i])) for i in {i for t in transacoes for i in t}}
    nivel = {c: s for c, s in nivel.items() if s >= min_suporte}
    frequentes = dict(nivel)
    k = 2
    while nivel and k <= max_itens:
        anteriores = list(nivel)
        candidatos = {a | b for i, a in enumerate(anteriores) for b in anteriores[i + 1:]
                      if len(a | b) == k}
        # poda do Apriori: todo subconjunto com k-1 itens também precisa ser frequente
        candidatos = [c for c in candidatos if all(c - {i} in nivel for i in c)]
        nivel = {c: s for c in candidatos if (s := suporte(c)) >= min_suporte}
        frequentes.update(nivel)
        k += 1
    return frequentes


def regras(frequentes, consequente, n):
    """Regras X -> consequente, a partir dos conjuntos frequentes que contêm o consequente."""
    alvo = frozenset([consequente])
    base = frequentes[alvo]
    todas = {}
    for c, s in frequentes.items():
        if consequente not in c or len(c) < 2:
            continue
        ant = c - alvo
        todas[ant] = dict(suporte=s, confianca=s / frequentes[ant])
    linhas = []
    for ant, r in todas.items():
        # melhor confiança entre as regras mais simples (subconjuntos próprios de X)
        sub = [v["confianca"] for a, v in todas.items() if a < ant]
        melhor_sub = max(sub) if sub else base
        linhas.append(dict(
            regra="{" + ", ".join(sorted(ant)) + "} -> {" + consequente + "}",
            itens=len(ant), campanhas_com_X=round(frequentes[ant] * n), casos=round(r["suporte"] * n),
            suporte=r["suporte"], confianca=r["confianca"], lift=r["confianca"] / base,
            redundante="sim" if r["confianca"] < melhor_sub + MELHORA_MINIMA else "nao"))
    out = pd.DataFrame(linhas)
    out = out[out.confianca >= MIN_CONFIANCA].sort_values(["confianca", "suporte"],
                                                          ascending=False)
    return out.round(3).reset_index(drop=True), base


# 1) Situação de cada time na rodada de corte e resultado final (2006-2025)
hist = pd.read_csv(PASTA / "brasileirao_limpo.csv", dtype={"ID": "string"})
hist = hist[(hist.temporada >= 2006) & (hist.temporada <= 2025)]
hist = hist[~hist["observacao"].fillna("").str.startswith("W.O.")]
x = linhas_por_time(hist)
final = tabela(x, PUNICOES)
final["rebaixado"] = (final.posicao > 20 - N_REBAIXADOS).astype(int)
t = tabela(x[x["rodata"] <= RODADA]).merge(final[["temporada", "time", "rebaixado"]],
                                           on=["temporada", "time"])
t["aproveitamento"] = t.pontos / (3 * t.jogos) * 100
t["gols_pro_jogo"] = t.gols_pro / t.jogos
t["gols_contra_jogo"] = t.gols_contra / t.jogos
t["pct_empates"] = t.empates / t.jogos * 100
t["aprov_casa"] = t.pts_casa / (3 * t.jogos_casa) * 100
t["aprov_fora"] = t.pts_fora / (3 * (t.jogos - t.jogos_casa)) * 100

# 2) Discretização em itens atributo=valor
TERCIS = {  # coluna: (atributo, rótulos do menor para o maior valor)
    "aproveitamento": ("aproveitamento", ["baixo", "medio", "alto"]),
    "gols_pro_jogo": ("ataque", ["fraco", "medio", "forte"]),
    "gols_contra_jogo": ("defesa", ["forte", "media", "fraca"]),   # mais gols sofridos = fraca
    "pct_empates": ("empates", ["poucos", "medios", "muitos"]),
    "aprov_casa": ("em_casa", ["fraco", "medio", "forte"]),
    "aprov_fora": ("fora", ["fraco", "medio", "forte"]),
}
itens = pd.DataFrame(index=t.index)
itens["posicao"] = np.where(t.posicao > 20 - N_REBAIXADOS, "Z4",
                            np.where(t.posicao > 12, "13a16", "1a12"))
itens["saldo"] = np.where(t.saldo < 0, "negativo", "zero_ou_positivo")
limites = []
for col, (attr, rotulos) in TERCIS.items():
    cat, bordas = pd.qcut(t[col], 3, labels=rotulos, retbins=True)
    itens[attr] = cat.astype(str)
    for rot, a, b in zip(rotulos, bordas[:-1], bordas[1:]):
        limites.append(dict(atributo=attr, valor=rot, coluna=col, de=round(a, 2), ate=round(b, 2),
                            campanhas=int((cat == rot).sum())))
itens["rebaixado"] = np.where(t.rebaixado == 1, "sim", "nao")
transacoes = [frozenset(f"{c}={v}" for c, v in linha.items()) for _, linha in itens.iterrows()]
n = len(transacoes)

pd.concat([t[["temporada", "time", "posicao", "pontos"]], itens.add_prefix("item_")], axis=1) \
    .to_csv(SAIDA / "regras_transacoes.csv", index=False, encoding="utf-8-sig")
limites = pd.DataFrame(limites)
limites.to_csv(SAIDA / "regras_discretizacao.csv", index=False, encoding="utf-8-sig")
print(f"Base: {n} campanhas, situação na rodada {RODADA}; "
      f"{(itens.rebaixado == 'sim').sum()} rebaixados ({(itens.rebaixado == 'sim').mean():.0%})")
print("\nLimites dos tercis:")
print(limites[["atributo", "valor", "de", "ate"]].to_string(index=False))

# 3) Apriori e regras
frequentes = apriori(transacoes, MIN_SUPORTE, MAX_ITENS)
print(f"\nConjuntos frequentes (suporte >= {MIN_SUPORTE:.0%}): {len(frequentes)}")

caiu, base = regras(frequentes, "rebaixado=sim", n)
caiu.to_csv(SAIDA / "regras_associacao.csv", index=False, encoding="utf-8-sig")
uteis = caiu[caiu.redundante == "nao"]
COLS = ["regra", "campanhas_com_X", "casos", "suporte", "confianca", "lift"]
print(f"\nRegras X -> rebaixado=sim (confiança >= {MIN_CONFIANCA:.0%}): {len(caiu)}, "
      f"{len(uteis)} não redundantes. Taxa geral de rebaixamento: {base:.0%}")
print(uteis[COLS].head(15).to_string(index=False))

z4 = itens.posicao == "Z4"
print(f"\nQuem estava no Z4 na rodada {RODADA}: {z4.sum()} campanhas, "
      f"{(z4 & (itens.rebaixado == 'nao')).sum()} escaparam. São poucos casos para minerar regras "
      f"de 'quem escapa' com o suporte mínimo de {MIN_SUPORTE:.0%}.")

# 4) Gráfico: suporte x confiança
fig, ax = plt.subplots(figsize=(8.5, 5.5))
sc = ax.scatter(caiu.suporte * 100, caiu.confianca * 100, c=caiu.lift, cmap="Reds", s=28,
                edgecolors=np.where(caiu.redundante == "nao", "black", "none"), linewidths=.7)
fig.colorbar(sc, ax=ax, label="lift")
ax.axhline(base * 100, color="gray", ls=":")
ax.text(caiu.suporte.min() * 100, base * 100 + 1.5, f"taxa geral de rebaixamento ({base:.0%})",
        color="gray", fontsize=9)
ax.set(xlabel="Suporte (% das campanhas)", ylabel="Confiança (% que caiu)",
       title=f"Regras X -> rebaixado (situação na rodada {RODADA}); borda preta = não redundante")
fig.tight_layout()
fig.savefig(SAIDA / "regras_associacao.png", dpi=130)
print("\nArquivos salvos em:", SAIDA.resolve())
