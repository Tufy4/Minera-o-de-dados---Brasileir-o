"""Coleta os resultados do Brasileirão 2026 (Série A) da Wikipedia e gera brasileirao_2026_jogos.csv.

A fonte é o wikitext da página '2026 Campeonato Brasileiro Série A', cuja matriz de resultados
é atualizada a partir da tabela da CBF. A matriz NÃO informa a rodada de cada jogo, então a
coluna 'rodata' fica em branco. Os nomes dos times seguem a grafia do histórico (sem acento).

Uso:  python coletar_2026.py [pasta_de_saida]
"""
import csv
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
SAIDA.mkdir(parents=True, exist_ok=True)

TITULO = "2026_Campeonato_Brasileiro_Série_A"
url = ("https://en.wikipedia.org/w/index.php?title=" + urllib.parse.quote(TITULO)
       + "&action=raw")
req = urllib.request.Request(url, headers={"User-Agent": "brasileirao-study/1.0"})
texto = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")

# Código usado na Wikipedia -> nome usado no histórico (brasileirao_limpo.csv)
NOMES = {
    "CAP": "Athletico-PR", "CAM": "Atletico-MG", "BAH": "Bahia", "BOT": "Botafogo-RJ",
    "CHA": "Chapecoense", "COR": "Corinthians", "CTB": "Coritiba", "CRU": "Cruzeiro",
    "FLA": "Flamengo", "FLU": "Fluminense", "GRE": "Gremio", "INT": "Internacional",
    "MIR": "Mirassol", "PAL": "Palmeiras", "RBB": "Bragantino", "REM": "Remo",
    "SAN": "Santos", "SPA": "Sao Paulo", "VAS": "Vasco", "VIT": "Vitoria",
}

data_atualizacao = re.search(r"\|update=([^\n|]+)", texto[texto.index("== Results =="):]).group(1)
jogos = []
for mand, vis, a, b in re.findall(r"\|match_([A-Z]{3})_([A-Z]{3})=\s*(?:\[\[[^\]|]*\|)?\s*(\d+)\s*[–-]\s*(\d+)", texto):
    jogos.append(dict(temporada=2026, rodata="", mandante=NOMES[mand], visitante=NOMES[vis],
                      mandante_Placar=int(a), visitante_Placar=int(b)))

# Confere com a classificação da própria página (vitórias/empates/derrotas/gols)
conf = {}
for cod, v, e, d, gp, gc in re.findall(
        r"\|win_([A-Z]{3})=(\d+)\s*\|draw_[A-Z]{3}=(\d+)\s*\|loss_[A-Z]{3}=(\d+)\s*"
        r"\|gf_[A-Z]{3}=(\d+)\s*\|ga_[A-Z]{3}=(\d+)", texto):
    conf[NOMES[cod]] = tuple(map(int, (v, e, d, gp, gc)))
calc = {n: [0, 0, 0, 0, 0] for n in NOMES.values()}
for j in jogos:
    for time, gf, ga in ((j["mandante"], j["mandante_Placar"], j["visitante_Placar"]),
                         (j["visitante"], j["visitante_Placar"], j["mandante_Placar"])):
        c = calc[time]
        c[0 if gf > ga else 1 if gf == ga else 2] += 1
        c[3] += gf; c[4] += ga
divergencias = [n for n in calc if tuple(calc[n]) != conf.get(n)]

with open(SAIDA / "brasileirao_2026_jogos.csv", "w", encoding="utf-8", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(jogos[0]))
    w.writeheader()
    w.writerows(jogos)

print(f"Fonte atualizada em: {data_atualizacao}")
print(f"Jogos lidos: {len(jogos)} (de 380 no total)")
print("Divergências com a classificação da página:", divergencias or "nenhuma")
