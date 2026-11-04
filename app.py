"""
RiskLab - interface de consulta de risco de crédito.

Como rodar:
    pip install streamlit
    streamlit run app.py

Requer, na mesma pasta: risklab_modelo.joblib, risklab_features.py e risklab_scoring.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from risklab_scoring import (
    COLUNAS, carregar_artefato, fatores_de_risco, pontuar_cliente, pontuar_lote,
)

st.set_page_config(page_title="RiskLab", page_icon="📊", layout="wide")


def pct(x, casas=1):
    return f"{x * 100:.{casas}f}%".replace(".", ",")


@st.cache_resource
def carrega_modelo():
    return carregar_artefato("risklab_modelo.joblib")


try:
    art = carrega_modelo()
except FileNotFoundError:
    st.error("Arquivo `risklab_modelo.joblib` não encontrado. Execute o notebook até o final para gerá-lo.")
    st.stop()

st.title("📊 RiskLab")
st.caption("Estimativa da probabilidade de inadimplência grave (90 dias ou mais) nos próximos 2 anos. "
           "Ferramenta de apoio à decisão do analista de crédito.")

aba_individual, aba_modelo = st.tabs(["Consulta individual", "Sobre o modelo"])

# ---------------------------------------------------------------------------
# 1) Consulta individual
# ---------------------------------------------------------------------------
with aba_individual:
    col_form, col_res = st.columns([1, 1.2], gap="large")

    with col_form:
        st.subheader("Dados do cliente")
        sem_renda = st.checkbox("Renda mensal não informada")

        with st.form("form_cliente"):
            renda = None
            if not sem_renda:
                renda = st.number_input("Renda mensal (R$)", min_value=1.0, max_value=1_000_000.0,
                                        value=5000.0, step=100.0)
            divida = st.number_input(
                "Pagamentos mensais de dívidas (R$)", min_value=0.0, max_value=1_000_000.0,
                value=1500.0, step=100.0,
                help="Soma mensal de parcelas, pensões e custos de vida. Com renda informada, "
                     "o modelo usa a razão dívida ÷ renda.")
            revolving = st.number_input(
                "Utilização do crédito rotativo (% do limite)", min_value=0.0, max_value=300.0,
                value=30.0, step=5.0,
                help="Quanto do limite de cartões e linhas sem garantia está em uso. 100% = limite totalmente usado.")
            idade = st.number_input("Idade", min_value=18, max_value=110, value=40, step=1)

            st.markdown("**Histórico de atrasos** (nº de vezes)")
            c1, c2, c3 = st.columns(3)
            a30 = c1.number_input("30-59 dias", min_value=0, max_value=20, value=0)
            a60 = c2.number_input("60-89 dias", min_value=0, max_value=20, value=0)
            a90 = c3.number_input("90+ dias", min_value=0, max_value=20, value=0)

            c4, c5 = st.columns(2)
            linhas = c4.number_input("Linhas de crédito abertas", min_value=0, max_value=60, value=6)
            imoveis = c5.number_input("Empréstimos imobiliários", min_value=0, max_value=30, value=1)

            c6, c7 = st.columns(2)
            deps = c6.number_input("Dependentes", min_value=0, max_value=15, value=0)
            deps_nao_inf = c7.checkbox("Dependentes não informado")

            enviado = st.form_submit_button("Calcular risco", type="primary")

    with col_res:
        st.subheader("Resultado")
        if not enviado:
            st.info("Preencha os dados e clique em **Calcular risco**.")
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

            m1, m2 = st.columns(2)
            m1.metric("Probabilidade de inadimplência", pct(r["probabilidade"]),
                      delta=(f"{r['probabilidade'] / base:.1f}x a média da base".replace(".", ",") if base else None),
                      delta_color="inverse")
            m2.metric("Faixa de risco", r["faixa"])

            msg = f"**Sugestão:** {r['sugestao']}  \n(limiar de recusa do modelo: {pct(art['limiar_recusa'])})"
            {"Baixo": st.success, "Moderado": st.info, "Alto": st.warning, "Muito alto": st.error}[r["faixa"]](msg)

            real = art.get("taxa_real_faixa", {}).get(r["faixa"])
            if real is not None:
                st.caption(f"Na base de teste, {pct(real)} dos clientes da faixa **{r['faixa']}** ficaram inadimplentes.")

            elevam, reduzem = fatores_de_risco(art, cliente)
            st.markdown("**O que mais pesa neste resultado** (comparado a um cliente típico)")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("Elevam o risco")
                if elevam:
                    for nome, v in elevam[:4]:
                        st.write(f"🔺 {nome} (≈ +{v:.1f} p.p.)".replace(".", ","))
                else:
                    st.write("Nenhum fator relevante.")
            with cb:
                st.markdown("Reduzem o risco")
                if reduzem:
                    for nome, v in reduzem[:4]:
                        st.write(f"🔻 {nome} (≈ {v:.1f} p.p.)".replace(".", ","))
                else:
                    st.write("Nenhum fator relevante.")
            st.caption("Cada fator é avaliado isoladamente (se fosse o de um cliente típico, a probabilidade "
                       "mudaria aproximadamente isso). Os efeitos não somam, pois os fatores interagem.")


# ---------------------------------------------------------------------------
# 3) Sobre o modelo
# ---------------------------------------------------------------------------
with aba_modelo:
    st.subheader("Sobre o modelo")
    st.write(f"**Algoritmo:** {art['modelo']} com probabilidades calibradas (calibração isotônica).  \n"
             f"**Base de treino:** Give Me Some Credit (Kaggle).  \n"
             f"**Limiar de recusa:** {pct(art['limiar_recusa'])}, definido minimizando o custo esperado "
             f"(custo relativo de aprovar um inadimplente x recusar um bom pagador = "
             f"{art['custos_relativos']['inadimplente']:.0f}:{art['custos_relativos']['bom_recusado']:.0f}, valores ilustrativos).")

    met = art.get("metricas_teste")
    if met:
        st.markdown(f"**Desempenho no conjunto de teste** ({art.get('n_teste', 0):,} clientes)".replace(",", "."))
        cols = st.columns(5)
        cols[0].metric("ROC-AUC", f"{met['ROC-AUC']:.3f}".replace(".", ","))
        cols[1].metric("Gini", f"{met['Gini']:.3f}".replace(".", ","))
        cols[2].metric("KS", f"{met['KS']:.3f}".replace(".", ","))
        cols[3].metric("PR-AUC", f"{met['PR-AUC']:.3f}".replace(".", ","))
        cols[4].metric("Brier", f"{met['Brier']:.4f}".replace(".", ","))
        ic = art.get("ic_auc_teste")
        if ic:
            st.caption(f"IC 95% da ROC-AUC: {ic[0]:.3f} a {ic[1]:.3f}".replace(".", ","))
    else:
        st.info("Métricas indisponíveis: reexecute o notebook atualizado para gerar o arquivo do modelo.")

    taxas = art.get("taxa_real_faixa")
    if taxas:
        st.markdown("**Inadimplência real observada por faixa de risco (teste)**")
        tabela = pd.DataFrame({"Inadimplência real": [taxas.get(f) for f in art["faixas"]]}, index=art["faixas"])
        st.bar_chart(tabela)

    st.markdown("""
**Como ler o resultado**
- A *probabilidade* é a chance estimada de o cliente atrasar 90 dias ou mais nos próximos 2 anos.
- A *faixa de risco* agrupa os clientes por posição no score (Baixo ≈ 50% menos arriscados; Muito alto ≈ 5% mais arriscados).
- A *sugestão* aplica o limiar de custo; é um apoio, não substitui a análise do analista.

**Limitações**
- O dataset contém apenas clientes que já receberam crédito (viés de seleção).
- Sem dimensão temporal: é preciso monitorar estabilidade e calibração ao longo do tempo.
- Custos do limiar são ilustrativos e devem ser substituídos pelos valores reais da empresa.
- Idade e dependentes em decisões automatizadas exigem revisão de compliance (por exemplo, LGPD).
""")
