"""
risklab_scoring.py
------------------
Lógica de consulta do RiskLab, separada da interface para poder ser testada
e reutilizada (Streamlit, API, scripts).

Requer risklab_modelo.joblib e risklab_features.py na mesma pasta.
"""
import joblib
import numpy as np
import pandas as pd

COLUNAS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Grupos de colunas usados para explicar "o que mais pesa" no risco de um cliente
GRUPOS_FATORES = {
    "Utilização do crédito rotativo": ["RevolvingUtilizationOfUnsecuredLines"],
    "Idade": ["age"],
    "Histórico de atrasos": [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ],
    "Endividamento": ["DebtRatio"],
    "Renda mensal": ["MonthlyIncome"],
    "Linhas de crédito abertas": ["NumberOfOpenCreditLinesAndLoans"],
    "Empréstimos imobiliários": ["NumberRealEstateLoansOrLines"],
    "Dependentes": ["NumberOfDependents"],
}

SUGESTAO_APROVAR = "Aprovar"
SUGESTAO_RECUSAR = "Recusar / encaminhar para análise"


def carregar_artefato(caminho="risklab_modelo.joblib"):
    return joblib.load(caminho)


def faixa_de_risco(art, p):
    """Faixa de risco (Baixo/Moderado/Alto/Muito alto) para uma probabilidade ou um array."""
    idx = np.digitize(p, art["cortes_faixas"])
    faixas = np.array(art["faixas"])
    return faixas[idx] if np.ndim(idx) else str(faixas[int(idx)])


def sugestao(art, p):
    return SUGESTAO_RECUSAR if p >= art["limiar_recusa"] else SUGESTAO_APROVAR


def prob_inadimplencia(art, df_raw):
    """Probabilidades para um DataFrame com as colunas brutas."""
    return art["pipeline"].predict_proba(df_raw[COLUNAS])[:, 1]


def pontuar_cliente(art, cliente: dict) -> dict:
    p = float(prob_inadimplencia(art, pd.DataFrame([cliente]))[0])
    return {
        "probabilidade": p,
        "faixa": faixa_de_risco(art, p),
        "sugestao": sugestao(art, p),
    }


def pontuar_lote(art, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia do DataFrame com probabilidade, faixa e sugestão."""
    faltando = [c for c in COLUNAS if c not in df_raw.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no arquivo: {faltando}")
    p = prob_inadimplencia(art, df_raw)
    saida = df_raw.copy()
    saida["prob_inadimplencia"] = p.round(4)
    saida["faixa_de_risco"] = faixa_de_risco(art, p)
    saida["sugestao"] = np.where(p >= art["limiar_recusa"], SUGESTAO_RECUSAR, SUGESTAO_APROVAR)
    return saida


def fatores_de_risco(art, cliente: dict):
    """
    Explicação simples por fator: quanto a probabilidade cairia (ou subiria) se o
    grupo de atributos fosse o de um cliente típico da base (medianas).

    Retorna (elevam, reduzem), listas de (fator, variacao_em_pontos_percentuais),
    ordenadas pela magnitude. Não é SHAP: é uma análise "um grupo por vez".
    """
    ref = art.get("referencia")
    if not ref:
        return [], []

    p0 = float(prob_inadimplencia(art, pd.DataFrame([cliente]))[0])
    contrafactuais = []
    for _, colunas in GRUPOS_FATORES.items():
        c = dict(cliente)
        for col in colunas:
            c[col] = ref[col]
        contrafactuais.append(c)

    p_ref = prob_inadimplencia(art, pd.DataFrame(contrafactuais))
    efeitos = [(nome, float((p0 - pr) * 100)) for nome, pr in zip(GRUPOS_FATORES, p_ref)]

    elevam = sorted([e for e in efeitos if e[1] > 0.05], key=lambda e: -e[1])
    reduzem = sorted([e for e in efeitos if e[1] < -0.05], key=lambda e: e[1])
    return elevam, reduzem
