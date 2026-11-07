from html import escape
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score  # noqa: F401  (mantido do original)

from risklab_scoring import (
    COLUNAS, carregar_artefato, fatores_de_risco, pontuar_cliente, pontuar_lote,  # noqa: F401
)

st.set_page_config(page_title="RiskLab", page_icon="📊", layout="wide",
                   menu_items={"Get help": None, "Report a bug": None, "About": None})

COR_FAIXA = {
    "Baixo": "#2F7D62",
    "Moderado": "#5B7F9E",
    "Alto": "#D98324",
    "Muito alto": "#B3261E",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --ink: #15212B;
  --muted: #5C6A75;
  --line: #DDE3E8;
  --bg: #F3F5F7;
  --surface: #FFFFFF;
  --accent: #0E5A66;
  --accent-dark: #0A4650;
  --good: #2F7D62;
  --bad: #B3261E;
}

html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp li, .stApp span.rl,
.stApp [data-baseweb="tab"] {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer { visibility: hidden; }

/* Barra superior (Fork, GitHub etc.): fundo sólido para o conteúdo não passar por baixo,
   e ícones totalmente opacos */
header, [data-testid="stHeader"] { background: #F3F5F7 !important; border-bottom: 1px solid #DDE3E8; }
header *, [data-testid="stToolbar"] *, [data-testid="stToolbarActions"] * { opacity: 1 !important; }
header a, header button, header span, header p, header svg,
[data-testid="stToolbar"] a, [data-testid="stToolbar"] button, [data-testid="stToolbar"] svg,
[data-testid="stToolbarActions"] a, [data-testid="stToolbarActions"] button,
[data-testid="stToolbarActions"] svg { color: #15212B !important; fill: currentColor !important; }
.block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 4rem; }

/* Cabeçalho */
.rl-head { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.rl-mark { width: 38px; height: 38px; border-radius: 9px; background: var(--accent);
  display: grid; place-items: center; flex: none; }
.rl-mark svg { display: block; }
.rl-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; margin: 0; color: var(--ink); }
.rl-sub { color: var(--muted); font-size: 0.95rem; max-width: 70ch; margin: 2px 0 0 52px; line-height: 1.5; }

/* Abas */
.stTabs [data-baseweb="tab-list"] { gap: 28px; border-bottom: 1px solid var(--line); }
[role="tablist"] [role="tab"] { padding: 10px 0; height: auto; font-weight: 500; }
[role="tablist"] [role="tab"],
[role="tablist"] [role="tab"] * { color: #4A5863 !important; opacity: 1 !important; }
[role="tablist"] [role="tab"]:hover,
[role="tablist"] [role="tab"]:hover * { color: #15212B !important; }
[role="tablist"] [role="tab"][aria-selected="true"],
[role="tablist"] [role="tab"][aria-selected="true"] * { color: #0E5A66 !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 2px; }

/* Títulos de seção */
.rl-h { font-size: 1.05rem; font-weight: 600; margin: 1.2rem 0 0.6rem; color: var(--ink); }
.rl-h:first-child { margin-top: 0.2rem; }
.rl-note { color: var(--muted); font-size: 0.82rem; line-height: 1.45; margin-top: 8px; }

/* Formulário */
[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 22px 22px 18px; }
.stApp label p { font-size: 0.86rem; font-weight: 500; color: var(--ink); }
.stApp input { background: #F8FAFB !important; color: var(--ink) !important; }
[data-testid="stNumberInput"] > div > div { border-radius: 8px; }
.stApp [data-testid="stCheckbox"] label p { font-weight: 400; }

/* Botão principal */
.stFormSubmitButton button, .stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {
  background: var(--accent); border: 1px solid var(--accent); color: #fff;
  border-radius: 8px; padding: 0.6rem 1.4rem; font-weight: 600; width: 100%;
}
.stFormSubmitButton button:hover, .stApp button[kind="primary"]:hover,
.stApp button[kind="primaryFormSubmit"]:hover { background: var(--accent-dark); border-color: var(--accent-dark); color: #fff; }
.stFormSubmitButton button:focus-visible { outline: 3px solid #9CC9D0; outline-offset: 2px; }

/* Estado vazio */
.rl-empty { border: 1.5px dashed #BAC5CD; border-radius: 12px; padding: 36px 26px; color: var(--muted);
  background: rgba(255,255,255,0.55); line-height: 1.55; }
.rl-empty b { color: var(--ink); }

/* Cartão de resultado */
.rl-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.rl-card-top { display: flex; justify-content: space-between; align-items: flex-start;
  padding: 22px 24px 18px; gap: 16px; }
.rl-lbl { color: var(--muted); font-size: 0.88rem; font-weight: 500; }
.rl-num { font-size: 3.6rem; font-weight: 700; letter-spacing: -0.035em; line-height: 1.05;
  font-variant-numeric: tabular-nums; margin-top: 4px; }
.rl-rel { color: var(--muted); font-size: 0.88rem; margin-top: 4px; }
.rl-chip { display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 600;
  font-size: 0.92rem; color: #fff; white-space: nowrap; }

/* Escala de faixas */
.rl-scale { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; padding: 0 24px 20px; }
.rl-seg { opacity: 0.4; }
.rl-seg.on { opacity: 1; }
.rl-seg-bar { height: 8px; border-radius: 4px; margin-bottom: 8px; }
.rl-seg-name { font-size: 0.85rem; font-weight: 500; color: var(--ink); }
.rl-seg.on .rl-seg-name { font-weight: 700; }
.rl-seg-rate { font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; }

/* Sugestão */
.rl-advice { border-top: 1px solid var(--line); padding: 16px 24px 18px; background: #FAFBFC; line-height: 1.5; }
.rl-advice small { display: block; color: var(--muted); font-size: 0.82rem; margin-top: 4px; }

/* Fatores */
.rl-fgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }
.rl-fbox { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.rl-fbox h4 { margin: 0 0 12px; font-size: 0.92rem; font-weight: 600; }
.rl-frow { margin-bottom: 12px; }
.rl-frow:last-child { margin-bottom: 0; }
.rl-fline { display: flex; justify-content: space-between; gap: 10px; font-size: 0.86rem; margin-bottom: 4px; }
.rl-fval { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rl-ftrack { height: 6px; background: #EAEEF1; border-radius: 3px; }
.rl-ffill { height: 6px; border-radius: 3px; }
.rl-fnone { color: var(--muted); font-size: 0.86rem; }

/* Aba "Sobre o modelo" */
.rl-stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 10px 0 6px; }
.rl-stat { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.rl-stat-v { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.rl-stat-k { color: var(--muted); font-size: 0.82rem; }
.rl-spec { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 6px 20px; }
.rl-spec-row { display: grid; grid-template-columns: 190px 1fr; gap: 14px; padding: 12px 0;
  border-bottom: 1px solid var(--line); font-size: 0.92rem; line-height: 1.5; }
.rl-spec-row:last-child { border-bottom: none; }
.rl-spec-k { color: var(--muted); }
.rl-hbar { display: grid; grid-template-columns: 100px 1fr 60px; align-items: center; gap: 12px;
  margin-bottom: 10px; font-size: 0.88rem; }
.rl-hbar-track { height: 14px; background: #EAEEF1; border-radius: 4px; }
.rl-hbar-fill { height: 14px; border-radius: 4px; }
.rl-hbar-v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.rl-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px; }
.rl-panel ul { margin: 0; padding-left: 1.1rem; }
.rl-panel li { margin-bottom: 8px; line-height: 1.5; font-size: 0.92rem; }

@media (max-width: 900px) {
  .rl-stats { grid-template-columns: repeat(2, 1fr); }
  .rl-fgrid { grid-template-columns: 1fr; }
  .rl-spec-row { grid-template-columns: 1fr; gap: 2px; }
  .rl-num { font-size: 2.8rem; }
  .rl-sub { margin-left: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Mude para True para esconder por completo a barra superior (Fork, GitHub, menu).
ESCONDER_BARRA_SUPERIOR = False
if ESCONDER_BARRA_SUPERIOR:
    st.markdown(
        "<style>header, [data-testid='stHeader'], [data-testid='stToolbar'], "
        "[data-testid='stToolbarActions'], [data-testid='stAppDeployButton'], "
        ".stAppDeployButton, .viewerBadge_container__r5tak { display: none !important; }</style>",
        unsafe_allow_html=True,
    )

def pct(x, casas=1):
    return f"{x * 100:.{casas}f}%".replace(".", ",")


def virgula(s):
    return str(s).replace(".", ",")


def html(*partes):
    """Junta trechos de HTML sem indentação (evita virar bloco de código no Markdown)."""
    st.markdown("".join(partes), unsafe_allow_html=True)


def cor_da(faixa):
    return COR_FAIXA.get(faixa, "#5C6A75")


def card_resultado(r, base, art):
    cor = cor_da(r["faixa"])
    rel = f"{r['probabilidade'] / base:.1f}x a média da base".replace(".", ",") if base else ""
    taxas = art.get("taxa_real_faixa", {})

    segmentos = []
    for f in art["faixas"]:
        ativo = " on" if f == r["faixa"] else ""
        real = taxas.get(f)
        legenda = f"{pct(real)} inadimplentes" if real is not None else "&nbsp;"
        segmentos.append(
            f'<div class="rl-seg{ativo}"><div class="rl-seg-bar" style="background:{cor_da(f)}"></div>'
            f'<div class="rl-seg-name">{escape(f)}</div><div class="rl-seg-rate">{legenda}</div></div>'
        )

    html(
        '<div class="rl-card">',
        '<div class="rl-card-top"><div>',
        '<div class="rl-lbl">Probabilidade de inadimplência</div>',
        f'<div class="rl-num" style="color:{cor}">{pct(r["probabilidade"])}</div>',
        f'<div class="rl-rel">{rel}</div></div>',
        f'<span class="rl-chip" style="background:{cor}">Risco {escape(r["faixa"].lower())}</span></div>',
        '<div class="rl-scale">', "".join(segmentos), "</div>",
        '<div class="rl-advice">',
        f'<b>Sugestão:</b> {escape(str(r["sugestao"]))}',
        f'<small>Limiar de recusa do modelo: {pct(art["limiar_recusa"])}. '
        'Os percentuais sob cada faixa são a inadimplência real na base de teste.</small>',
        "</div></div>",
    )


def bloco_fatores(titulo, lista, cor, sinal):
    if not lista:
        corpo = '<div class="rl-fnone">Nenhum fator relevante.</div>'
    else:
        maximo = max(abs(v) for _, v in lista) or 1
        linhas = []
        for nome, v in lista:
            largura = max(6, abs(v) / maximo * 100)
            valor = virgula(f"{sinal}{abs(v):.1f} p.p.")
            linhas.append(
                '<div class="rl-frow"><div class="rl-fline">'
                f'<span>{escape(str(nome))}</span><span class="rl-fval" style="color:{cor}">{valor}</span></div>'
                f'<div class="rl-ftrack"><div class="rl-ffill" style="width:{largura:.0f}%;background:{cor}"></div></div></div>'
            )
        corpo = "".join(linhas)
    return f'<div class="rl-fbox"><h4>{titulo}</h4>{corpo}</div>'

@st.cache_resource
def carrega_modelo():
    return carregar_artefato("risklab_modelo.joblib")


try:
    art = carrega_modelo()
except FileNotFoundError:
    st.error("Arquivo `risklab_modelo.joblib` não encontrado. Execute o notebook até o final para gerá-lo.")
    st.stop()
    
html(
    '<div class="rl-head"><div class="rl-mark">',
    '<svg width="22" height="22" viewBox="0 0 22 22" fill="none">',
    '<rect x="3" y="12" width="4" height="7" rx="1" fill="#fff" opacity=".6"/>',
    '<rect x="9" y="7" width="4" height="12" rx="1" fill="#fff" opacity=".8"/>',
    '<rect x="15" y="3" width="4" height="16" rx="1" fill="#fff"/></svg></div>',
    '<h1 class="rl-title">RiskLab</h1></div>',
    '<p class="rl-sub">Estimativa da probabilidade de inadimplência grave (90 dias ou mais) nos próximos 2 anos. '
    "Ferramenta de apoio à decisão do analista de crédito.</p>",
)
st.write("")

aba_individual, aba_modelo = st.tabs(["Consulta individual", "Sobre o modelo"])


with aba_individual:
    st.write("")
    col_form, col_res = st.columns([1, 1.2], gap="large")

    with col_form:
        html('<div class="rl-h">Dados do cliente</div>')
        sem_renda = st.checkbox("Renda mensal não informada")

        with st.form("form_cliente"):
            renda = None
            if not sem_renda:
                renda = st.number_input("Renda mensal (R$)", min_value=0.0, max_value=1_000_000.0,
                                        value=0.0, step=100.0)
            divida = st.number_input(
                "Pagamentos mensais de dívidas (R$)", min_value=0.0, max_value=1_000_000.0,
                value=0.0, step=100.0,
                help="Soma mensal de parcelas, pensões e custos de vida. Com renda informada, "
                     "o modelo usa a razão dívida ÷ renda.")
            revolving = st.number_input(
                "Utilização do crédito rotativo (% do limite)", min_value=0.0, max_value=300.0,
                value=0.0, step=5.0,
                help="Quanto do limite de cartões e linhas sem garantia está em uso. 100% = limite totalmente usado.")
            idade = st.number_input("Idade", min_value=18, max_value=110, value=0, step=1)

            html('<div class="rl-h">Histórico de atrasos (nº de vezes)</div>')
            c1, c2, c3 = st.columns(3)
            a30 = c1.number_input("30-59 dias", min_value=0, max_value=20, value=0)
            a60 = c2.number_input("60-89 dias", min_value=0, max_value=20, value=0)
            a90 = c3.number_input("90+ dias", min_value=0, max_value=20, value=0)

            html('<div class="rl-h">Crédito e família</div>')
            c4, c5 = st.columns(2)
            linhas = c4.number_input("Linhas de crédito abertas", min_value=0, max_value=60, value=0)
            imoveis = c5.number_input("Empréstimos imobiliários", min_value=0, max_value=30, value=0)

            c6, c7 = st.columns(2)
            deps = c6.number_input("Dependentes", min_value=0, max_value=15, value=0)
            deps_nao_inf = c7.checkbox("Dependentes não informado")

            enviado = st.form_submit_button("Calcular risco", type="primary")

    with col_res:
        html('<div class="rl-h">Resultado</div>')
        if not enviado:
            html('<div class="rl-empty"><b>Nenhuma consulta ainda.</b><br>'
                 "Preencha os dados do cliente e clique em <b>Calcular risco</b> para ver a probabilidade, "
                 "a faixa de risco e os fatores que mais pesam.</div>")
        else:
            # Monta o registro bruto no formato do dataset original.
            # Sem renda informada, o DebtRatio original guarda a dívida absoluta.
            cliente = {
                "RevolvingUtilizationOfUnsecuredLines": revolving / 100,
                "age": idade,
                "NumberOfTime30-59DaysPastDueNotWorse": a30,
                "DebtRatio": divida if sem_renda else divida / renda,
                "MonthlyIncome": np.nan if sem_renda else renda,
                "NumberOfOpenCreditLinesAndLoans": linhas,
                "NumberOfTimes90DaysLate": a90,
                "NumberRealEstateLoansOrLines": imoveis,
                "NumberOfTime60-89DaysPastDueNotWorse": a60,
                "NumberOfDependents": np.nan if deps_nao_inf else deps,
            }
            r = pontuar_cliente(art, cliente)
            base = art.get("taxa_base")

            card_resultado(r, base, art)

            elevam, reduzem = fatores_de_risco(art, cliente)
            html('<div class="rl-h" style="margin-top:1.6rem">O que mais pesa neste resultado</div>',
                 '<div class="rl-note" style="margin-top:-4px">Comparado a um cliente típico.</div>',
                 '<div class="rl-fgrid">',
                 bloco_fatores("Elevam o risco", elevam[:4], "#B3261E", "+"),
                 bloco_fatores("Reduzem o risco", reduzem[:4], "#2F7D62", "−"),
                 "</div>",
                 '<div class="rl-note">Cada fator é avaliado isoladamente (se fosse o de um cliente típico, '
                 "a probabilidade mudaria aproximadamente isso). Os efeitos não somam, pois os fatores interagem.</div>")

with aba_modelo:
    st.write("")
    html('<div class="rl-h">Sobre o modelo</div>')
    cr = art["custos_relativos"]
    html(
        '<div class="rl-spec">',
        f'<div class="rl-spec-row"><div class="rl-spec-k">Algoritmo</div>'
        f'<div>{escape(str(art["modelo"]))}, com probabilidades calibradas (calibração isotônica).</div></div>',
        '<div class="rl-spec-row"><div class="rl-spec-k">Base de treino</div>'
        "<div>Give Me Some Credit (Kaggle).</div></div>",
        f'<div class="rl-spec-row"><div class="rl-spec-k">Limiar de recusa</div>'
        f'<div>{pct(art["limiar_recusa"])}, definido minimizando o custo esperado. Custo relativo de aprovar '
        f'um inadimplente x recusar um bom pagador = {cr["inadimplente"]:.0f}:{cr["bom_recusado"]:.0f} '
        "(valores ilustrativos).</div></div>",
        "</div>",
    )

    met = art.get("metricas_teste")
    if met:
        n = f"{art.get('n_teste', 0):,}".replace(",", ".")
        html(f'<div class="rl-h" style="margin-top:1.6rem">Desempenho no conjunto de teste ({n} clientes)</div>')
        itens = [("ROC-AUC", 3), ("Gini", 3), ("KS", 3), ("PR-AUC", 3), ("Brier", 4)]
        cartoes = "".join(
            f'<div class="rl-stat"><div class="rl-stat-v">{virgula(f"{met[k]:.{d}f}")}</div>'
            f'<div class="rl-stat-k">{k}</div></div>'
            for k, d in itens
        )
        html(f'<div class="rl-stats">{cartoes}</div>')
        ic = art.get("ic_auc_teste")
        if ic:
            html(f'<div class="rl-note">{virgula(f"IC 95% da ROC-AUC: {ic[0]:.3f} a {ic[1]:.3f}")}</div>')
    else:
        st.info("Métricas indisponíveis: reexecute o notebook atualizado para gerar o arquivo do modelo.")

    taxas = art.get("taxa_real_faixa")
    if taxas:
        html('<div class="rl-h" style="margin-top:1.6rem">Inadimplência real observada por faixa de risco (teste)</div>')
        valores = [taxas.get(f) or 0 for f in art["faixas"]]
        topo = max(valores) or 1
        barras = "".join(
            f'<div class="rl-hbar"><span>{escape(f)}</span>'
            f'<div class="rl-hbar-track"><div class="rl-hbar-fill" style="width:{max(1, v / topo * 100):.0f}%;'
            f'background:{cor_da(f)}"></div></div><span class="rl-hbar-v">{pct(v)}</span></div>'
            for f, v in zip(art["faixas"], valores)
        )
        html(f'<div class="rl-panel">{barras}</div>')

    st.write("")
    ca, cb = st.columns(2, gap="large")
    with ca:
        html(
            '<div class="rl-panel"><div class="rl-h">Como ler o resultado</div><ul>',
            "<li>A <i>probabilidade</i> é a chance estimada de o cliente atrasar 90 dias ou mais nos próximos 2 anos.</li>",
            "<li>A <i>faixa de risco</i> agrupa os clientes por posição no score "
            "(Baixo ≈ 50% menos arriscados; Muito alto ≈ 5% mais arriscados).</li>",
            "<li>A <i>sugestão</i> aplica o limiar de custo; é um apoio, não substitui a análise do analista.</li>",
            "</ul></div>",
        )
    with cb:
        html(
            '<div class="rl-panel"><div class="rl-h">Limitações</div><ul>',
            "<li>O dataset contém apenas clientes que já receberam crédito (viés de seleção).</li>",
            "<li>Sem dimensão temporal: é preciso monitorar estabilidade e calibração ao longo do tempo.</li>",
            "<li>Custos do limiar são ilustrativos e devem ser substituídos pelos valores reais da empresa.</li>",
            "<li>Idade e dependentes em decisões automatizadas exigem revisão de compliance (por exemplo, LGPD).</li>",
            "</ul></div>",
        )
