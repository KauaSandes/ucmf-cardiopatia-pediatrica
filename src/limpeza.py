"""Limpeza do conjunto UCMF (etapa definida na Avaliação A1).

Lê data/UCMF_raw.csv e grava data/UCMF_limpo.csv e results/limpeza.json (contagens de cada passo).
As correções aqui não dependem de estatísticas da amostra que envolvam o alvo e podem ser aplicadas
antes da divisão treino/teste. As exceções (cercas IQR por idade/sexo) usam apenas atributos de entrada
e são discutidas no relatório.

Executar na raiz do repositório:  python src/limpeza.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ALVO = "NORMAL X ANORMAL"
NIV = ["Normal", "Pré-hipertensão", "HAS estágio 1", "HAS estágio 2"]
IDX_P50 = 3  # coluna do percentil 50 de estatura (percentil individual não disponível)


# ---------------------------------------------------------------- utilidades
def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def parse_data(x):
    """dd/mm/aa, dd/mm/aaaa ou número serial do Excel -> Timestamp; demais formatos -> NaT."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if re.fullmatch(r"\d{5}", s):  # serial do Excel (dias desde 1899-12-30)
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})", s)
    if not m:
        return pd.NaT
    d, mo, a = map(int, m.groups())
    if a < 100:
        a = 2000 + a if a <= 12 else 1900 + a
    try:
        return pd.Timestamp(a, mo, d)
    except ValueError:  # ex.: 31/04, 29/02 em ano não bissexto
        return pd.NaT


def parse_fc(x):
    """FC numérica; faixas registradas como '140-160' ou '156 a 168' viram o ponto médio."""
    if pd.isna(x):
        return np.nan
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(x))]
    return np.mean(nums) if nums else np.nan


def nivel(v, t, j, adulto_cut):
    if pd.isna(v):
        return np.nan
    p90, p95, p99 = t["90th"][j], t["95th"][j], t["99th"][j]
    if v > p99 + 5:
        return 3
    if v >= p95:
        return 2
    if v >= p90 or v >= adulto_cut:
        return 1
    return 0


def classificar_ppa(row, bp):
    """PPA pelas tabelas pediátricas (NHBPEP, 2004): HAS se PAS OU PAD elevada; >= 18 anos, critério adulto."""
    pas, pad, idd, sx = row["PA SISTOLICA"], row["PA DIASTOLICA"], row["IDADE"], row["SEXO"]
    if pd.isna(pas) and pd.isna(pad):
        return "Não aferida"
    if pd.isna(idd) or pd.isna(sx) or idd < 1:
        return "Não calculável"
    if idd >= 18:
        n = max(0 if pd.isna(pas) or pas < 120 else 1 if pas < 140 else 2 if pas < 160 else 3,
                0 if pd.isna(pad) or pad < 80 else 1 if pad < 90 else 2 if pad < 100 else 3)
        return NIV[n]
    t = bp[sx][str(int(idd))]
    n = max(nivel(pas, {k: v[:7] for k, v in t.items()}, IDX_P50, 120),
            nivel(pad, {k: v[7:] for k, v in t.items()}, IDX_P50, 80))
    return NIV[int(n)] if not pd.isna(n) else "Não calculável"


# ---------------------------------------------------------------- limpeza
def limpar(raw, bp):
    """Aplica os passos de limpeza da A1 e devolve (base limpa, contagens)."""
    df = raw.copy()
    log = {"n_bruto": int(len(df)), "p_bruto": int(df.shape[1])}

    # registros vazios e duplicados
    clin = ["Peso", "Altura", "IMC", "Atendimento", "DN", "IDADE", "Convenio", "PULSOS", "PA SISTOLICA",
            "PA DIASTOLICA", ALVO, "B2", "SOPRO", "FC", "HDA 1", "HDA2", "MOTIVO1", "MOTIVO2"]
    vazio = df[clin].replace(0, np.nan).isna().all(axis=1)
    log["vazios"] = int(vazio.sum())
    df = df[~vazio]
    log["duplicatas"] = int(df.drop(columns="ID").duplicated().sum())
    df = df.drop_duplicates(subset=df.columns.drop("ID"))

    # padronização léxica
    df["SEXO"] = df["SEXO"].str.strip().str.lower().map({"m": "M", "masculino": "M", "f": "F", "feminino": "F"})
    df[ALVO] = df[ALVO].str.strip().str.lower().map({"normal": "Normal", "normais": "Normal", "anormal": "Anormal"})
    df["PULSOS"] = df["PULSOS"].str.strip().str.capitalize().replace({"Femorais diminuidos": "Femorais diminuídos"})
    df["SOPRO"] = df["SOPRO"].str.strip().map(lambda s: s if pd.isna(s) else sem_acento(s).lower()).map(
        {"ausente": "Ausente", "sistolico": "Sistólico", "continuo": "Contínuo", "diastolico": "Diastólico",
         "sistolico e diastolico": "Sistólico e diastólico"})
    df["B2"] = df["B2"].str.strip()
    df["Convenio"] = df["Convenio"].str.strip().str.upper()

    # alvo ausente
    log["alvo_ausente"] = int(df[ALVO].isna().sum())
    df = df[df[ALVO].notna()]

    # datas e idade (a IDADE original equivale a dias/352)
    A = df["Atendimento"].map(parse_data)
    D = df["DN"].map(parse_data)
    A[A.notna() & ((A < "2003-01-01") | (A > "2010-12-31"))] = pd.NaT
    dias = (A - D).dt.days
    divisor = float((dias / df["IDADE"])[(dias > 30) & df["IDADE"].gt(0)].median())
    idade = dias / 365.25
    sem_data = idade.isna() & df["IDADE"].notna()
    idade[sem_data] = df.loc[sem_data, "IDADE"] * divisor / 365.25
    log["idade_divisor_original"] = round(divisor, 1)
    log["idade_fora_0_19"] = int(((idade < 0) | (idade >= 20)).sum())
    idade[(idade < 0) | (idade >= 20)] = np.nan
    df["IDADE"] = idade.round(2)

    # antropometria
    for c in ["Peso", "Altura"]:
        df.loc[df[c] <= 0, c] = np.nan
    df.loc[(df["Peso"] < 0.5) | (df["Peso"] > 150), "Peso"] = np.nan
    df.loc[(df["Altura"] < 35) | (df["Altura"] > 200), "Altura"] = np.nan
    faixa = np.floor(df["IDADE"])
    for c in ["Peso", "Altura"]:
        g = df.groupby([faixa, df["SEXO"]])[c]
        q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
        df.loc[(df[c] < q1 - 3 * (q3 - q1)) | (df[c] > q3 + 3 * (q3 - q1)), c] = np.nan
    imc = df["Peso"] / (df["Altura"] / 100) ** 2
    df.loc[imc.notna() & ((imc < 8) | (imc > 50)), ["Peso", "Altura"]] = np.nan
    df["IMC"] = (df["Peso"] / (df["Altura"] / 100) ** 2).round(1)

    # pressão arterial e PPA
    df.loc[(df["PA SISTOLICA"] < 60) | (df["PA SISTOLICA"] > 200), "PA SISTOLICA"] = np.nan
    df.loc[(df["PA DIASTOLICA"] < 30) | (df["PA DIASTOLICA"] > 130), "PA DIASTOLICA"] = np.nan
    df.loc[df["PA DIASTOLICA"] >= df["PA SISTOLICA"], ["PA SISTOLICA", "PA DIASTOLICA"]] = np.nan
    df["PPA"] = df.apply(classificar_ppa, axis=1, bp=bp)

    # frequência cardíaca
    fc = df["FC"].map(parse_fc)
    fc[(fc < 40) | (fc > 230)] = np.nan
    df["FC"] = fc

    # ausência informativa em histórico e motivo
    for c in ["HDA 1", "MOTIVO1", "MOTIVO2"]:
        df[c] = df[c].fillna("Não informado")

    # contradições (mesmos atributos, classes diferentes): exclui todas
    feat = [c for c in df.columns if c not in ("ID", ALVO)]
    g = df.groupby(feat, dropna=False)[ALVO].transform("nunique")
    log["contraditorios"] = int((g > 1).sum())
    df = df[g <= 1]

    log["n_limpo"] = int(len(df))
    log["alvo_limpo"] = df[ALVO].value_counts().to_dict()
    return df, log


if __name__ == "__main__":
    raw = pd.read_csv(RAIZ / "data" / "UCMF_raw.csv", encoding="utf-8")
    bp = json.loads((RAIZ / "data" / "tabela_pa.json").read_text(encoding="utf-8"))["sexo_idade"]
    df, log = limpar(raw, bp)
    df.to_csv(RAIZ / "data" / "UCMF_limpo.csv", index=False, encoding="utf-8")
    (RAIZ / "results").mkdir(exist_ok=True)
    (RAIZ / "results" / "limpeza.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False))
