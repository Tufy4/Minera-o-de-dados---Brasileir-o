
import csv
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import kagglehub

SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)

# 1) Download do dataset (fica em cache em ~/.cache/kagglehub)
pasta = Path(kagglehub.dataset_download("adaoduque/campeonato-brasileiro-de-futebol"))
with open(pasta / "campeonato-brasileiro-full.csv", encoding="utf-8", newline="") as f:
    jogos = list(csv.DictReader(f))

# 2) Temporada = ano da data; a de 2020 terminou em fev/2021 (pandemia)
for r in jogos:
    d = datetime.strptime(r["data"], "%d/%m/%Y")
    r["temporada"] = d.year - 1 if (d.year == 2021 and d.month <= 4) else d.year

# 3) Limpeza: mantém só as colunas escolhidas (remove data, hora, estados, arrecadação,
#    formações, técnicos e arena)
COLS = ["ID", "temporada", "rodata", "mandante", "visitante", "vencedor",
        "mandante_Placar", "visitante_Placar", "observacao"]
limpo = [{**{c: r[c] for c in COLS[:-1]}, "observacao": ""} for r in jogos]

# 4) Chapecoense x Atlético-MG (2016, rodada 38): W.O. duplo, derrota 0x3 para cada clube
limpo.append({"ID": "", "temporada": "2016", "rodata": "38", "mandante": "Chapecoense",
              "visitante": "Atletico-MG", "vencedor": "-", "mandante_Placar": "0",
              "visitante_Placar": "0",
              "observacao": "W.O. duplo: derrota 0x3 para cada clube, sem pontos"})

with open(SAIDA / "brasileirao_limpo.csv", "w", encoding="utf-8", newline="") as o:
    w = csv.DictWriter(o, fieldnames=COLS)
    w.writeheader()
    w.writerows(limpo)

# 5) Classificação por temporada
tab = defaultdict(lambda: defaultdict(lambda: dict(J=0, V=0, E=0, D=0, GP=0, GC=0)))
for r in limpo:
    t = int(r["temporada"])
    gm, gv = int(r["mandante_Placar"]), int(r["visitante_Placar"])
    if r["observacao"].startswith("W.O. duplo"):
        for time in (r["mandante"], r["visitante"]):
            s = tab[t][time]
            s["J"] += 1; s["D"] += 1; s["GC"] += 3
        continue
    for time, gf, ga in ((r["mandante"], gm, gv), (r["visitante"], gv, gm)):
        s = tab[t][time]
        s["J"] += 1; s["GP"] += gf; s["GC"] += ga
        s["V" if gf > ga else "E" if gf == ga else "D"] += 1

# 6) 10 últimos de cada temporada + aproveitamento
ultimos = []
for t in sorted(tab):
    linhas = []
    for time, s in tab[t].items():
        s = dict(s, time=time)
        s["PTS"] = 3 * s["V"] + s["E"]
        s["SG"] = s["GP"] - s["GC"]
        linhas.append(s)
    # desempate: pontos, vitórias, saldo de gols, gols pró
    linhas.sort(key=lambda s: (-s["PTS"], -s["V"], -s["SG"], -s["GP"]))
    n = len(linhas)
    for pos, s in enumerate(linhas, 1):
        if pos > n - 10:
            ultimos.append(dict(
                temporada=t, posicao=pos, time=s["time"], pontos=s["PTS"], jogos=s["J"],
                vitorias=s["V"], empates=s["E"], derrotas=s["D"], gols_pro=s["GP"],
                gols_contra=s["GC"], saldo_gols=s["SG"],
                aproveitamento=round(s["PTS"] / (3 * s["J"]) * 100, 2)))

with open(SAIDA / "ultimos10_por_temporada.csv", "w", encoding="utf-8-sig", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(ultimos[0]))
    w.writeheader()
    w.writerows(ultimos)

# 7) Média e mediana do aproveitamento (por temporada e geral)
por_ano = defaultdict(list)
for r in ultimos:
    por_ano[r["temporada"]].append(r["aproveitamento"])
resumo = [dict(temporada=t, media=round(st.mean(v), 2), mediana=round(st.median(v), 2),
               minimo=min(v), maximo=max(v)) for t, v in por_ano.items()]
todos = [r["aproveitamento"] for r in ultimos]
resumo.append(dict(temporada="GERAL", media=round(st.mean(todos), 2),
                   mediana=round(st.median(todos), 2), minimo=min(todos), maximo=max(todos)))

with open(SAIDA / "aproveitamento_resumo.csv", "w", encoding="utf-8-sig", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(resumo[0]))
    w.writeheader()
    w.writerows(resumo)

print("Arquivos salvos em:", SAIDA.resolve())
