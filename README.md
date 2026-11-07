# RiskLab | Análise de Risco de Crédito

**Uma solução de Machine Learning para estimativa da probabilidade de inadimplência.**

🌐 **Acesse a aplicação:** [risklabanalytics.streamlit.app](https://risklabanalytics.streamlit.app/)

## Sobre o projeto

O RiskLab é um projeto de Ciência de Dados e Machine Learning desenvolvido com o objetivo de estimar a probabilidade de inadimplência de clientes em um horizonte de dois anos.

A solução busca apoiar analistas de crédito na avaliação de riscos, transformando dados históricos em estimativas que podem auxiliar o processo de tomada de decisão.

## Objetivos

- Explorar e compreender os dados relacionados ao risco de crédito.
- Identificar padrões associados à inadimplência.
- Preparar e tratar os dados para modelagem.
- Desenvolver e avaliar modelos preditivos.
- Interpretar os resultados e as variáveis relevantes.
- Disponibilizar as previsões por meio de uma aplicação interativa.

## Metodologia

O desenvolvimento foi conduzido seguindo etapas inspiradas na metodologia CRISP-DM:

1. **Entendimento do negócio:** definição do problema e dos objetivos da análise.
2. **Entendimento dos dados:** exploração da base utilizando SQL e Pandas.
3. **Preparação dos dados:** tratamento de valores ausentes, inconsistências e valores extremos, além da engenharia de atributos.
4. **Modelagem:** treinamento e comparação de diferentes algoritmos de Machine Learning.
5. **Avaliação:** análise do desempenho dos modelos por meio de métricas de classificação.
6. **Implementação:** desenvolvimento de uma interface interativa utilizando Streamlit.

## Tecnologias utilizadas

- **Python:** desenvolvimento e processamento dos dados.
- **Pandas e NumPy:** manipulação e análise de dados.
- **SQL e SQLite:** consultas e exploração da base de dados.
- **Scikit-learn:** treinamento e avaliação dos modelos.
- **Streamlit:** desenvolvimento da aplicação web.
- **Joblib:** armazenamento e carregamento do modelo treinado.

## Funcionalidades

- Estimativa individual da probabilidade de inadimplência.
- Análise de múltiplos clientes por meio de arquivos CSV.
- Classificação dos níveis de risco.
- Apresentação de informações para apoiar a interpretação dos resultados.
- Interface interativa para consulta das previsões.

## Base de dados

O projeto utiliza a base **Give Me Some Credit**, disponibilizada no Kaggle.

O conjunto de dados contém informações financeiras e comportamentais de clientes, tendo como variável-alvo a ocorrência de inadimplência grave nos dois anos seguintes.

## Como executar localmente

Clone o repositório:

```bash
git clone https://github.com/dudadelrei1/RiskLab.git
```

Entre na pasta do projeto:

```bash
cd RiskLab
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

## Aplicação online

Acesse o RiskLab diretamente pelo Streamlit Community Cloud:

🔗 https://risklabanalytics.streamlit.app/

## Observação

O RiskLab possui finalidade educacional e demonstrativa. As estimativas apresentadas são ferramentas de apoio à análise de risco e não representam decisões definitivas de concessão ou recusa de crédito.

---

**Desenvolvido como projeto pessoal de Ciência de Dados e Machine Learning.**
