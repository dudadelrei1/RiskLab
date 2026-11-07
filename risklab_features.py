import numpy as np
import pandas as pd

# Códigos especiais (não são contagens reais) encontrados nas 3 colunas de atraso
CODIGOS_ESPECIAIS = (96, 98)

# Tetos definidos a partir do EDA (não aprendidos dos dados):
#  - Revolving: p99 = 1,09; acima de 2 os valores são implausíveis (limite ~0)
#  - DebtRatio (quando é de fato uma razão): acima de 10 já é renda declarada irrisória
TETO_REVOLVING = 2.0
TETO_DEBTRATIO = 10.0

COLS_ATRASO = [
    "numberoftime30-59dayspastduenotworse",
    "numberoftime60-89dayspastduenotworse",
    "numberoftimes90dayslate",
]

FEATURES = [
    "revolving",
    "revolving_acima_1",
    "age",
    "atrasos_30_59",
    "atrasos_60_89",
    "atrasos_90",
    "total_atrasos",
    "atraso_ponderado",
    "atraso_codigo_especial",
    "debtratio",
    "log_divida_mensal",
    "log_renda",
    "renda_ausente",
    "log_renda_por_dependente",
    "dependentes",
    "dependentes_ausentes",
    "linhas_credito_abertas",
    "emprestimos_imobiliarios",
]


def prepara_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = d.columns.str.strip().str.lower()

    # --- Atrasos: 96/98 são códigos especiais -> NaN + flag ---------------
    atrasos = d[COLS_ATRASO].astype(float)
    especial = atrasos.isin(CODIGOS_ESPECIAIS).any(axis=1)
    atrasos = atrasos.mask(atrasos.isin(CODIGOS_ESPECIAIS))

    n30, n60, n90 = (atrasos[c] for c in COLS_ATRASO)
    total_atrasos = (n30.fillna(0) + n60.fillna(0) + n90.fillna(0)).mask(especial)
    # atrasos mais longos pesam mais
    atraso_ponderado = (n30.fillna(0) + 2 * n60.fillna(0) + 3 * n90.fillna(0)).mask(especial)

    # --- Renda e dependentes ---------------------------------------------
    renda = d["monthlyincome"].astype(float)
    renda_ausente = renda.isna()
    deps = d["numberofdependents"].astype(float)

    # --- DebtRatio: só é razão quando há renda ----------------------------
    # Achado do EDA: sem renda informada, a coluna guarda a dívida ABSOLUTA.
    # Reconstruímos a dívida mensal para todos os clientes.
    debt = d["debtratio"].astype(float)
    divida_mensal = pd.Series(
        np.where(renda_ausente, debt, debt * renda), index=d.index
    )
    debtratio = debt.where(~renda_ausente).clip(lower=0, upper=TETO_DEBTRATIO)

    # --- Revolving: teto + flag de utilização acima do limite -------------
    rev = d["revolvingutilizationofunsecuredlines"].astype(float)

    # --- Idade 0 é erro de cadastro -> NaN --------------------------------
    idade = d["age"].astype(float)
    idade = idade.mask(idade <= 0)

    out = pd.DataFrame(index=d.index)
    out["revolving"] = rev.clip(lower=0, upper=TETO_REVOLVING)
    out["revolving_acima_1"] = (rev > 1).astype(int)
    out["age"] = idade
    out["atrasos_30_59"] = n30
    out["atrasos_60_89"] = n60
    out["atrasos_90"] = n90
    out["total_atrasos"] = total_atrasos
    out["atraso_ponderado"] = atraso_ponderado
    out["atraso_codigo_especial"] = especial.astype(int)
    out["debtratio"] = debtratio
    out["log_divida_mensal"] = np.log1p(divida_mensal.clip(lower=0))
    out["log_renda"] = np.log1p(renda.clip(lower=0))
    out["renda_ausente"] = renda_ausente.astype(int)
    out["log_renda_por_dependente"] = np.log1p(
        (renda / (deps.fillna(0) + 1)).clip(lower=0)
    )
    out["dependentes"] = deps
    out["dependentes_ausentes"] = deps.isna().astype(int)
    out["linhas_credito_abertas"] = d["numberofopencreditlinesandloans"].astype(float)
    out["emprestimos_imobiliarios"] = d["numberrealestateloansorlines"].astype(float)

    return out[FEATURES]
