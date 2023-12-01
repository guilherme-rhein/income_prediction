<h1 align="center">
    💸 Previsão de Renda 💸</a>
</h1>

<p align="center"> O objetivo do projeto é um modelo baseado em dados, com os estudos realizados para prever valores de renda. </p>

---
<h3 align="center">
    🎈 Experimente Aqui: <a href="https://income-prediction.streamlit.app/">Modelo de Previsão de Renda 🎈 </a>
</h3>

---

https://github.com/guilherme-rhein/income_prediction/assets/88910779/7fc0df49-d519-4591-a00e-782fcd069805


## Base do Projeto ⚖️

Uma instituição financeira está empenhada em aprimorar sua compreensão acerca do perfil de renda de seus clientes,
com o intuito de otimizar a definição dos limites de cartões de crédito, sem a necessidade de solicitar comprovantes
de renda ou documentos que possam impactar a experiência do cliente.

Para alcançar esse objetivo, foi disponibilizado uma base com dados com diferentes variáveis para que fosse possível
treinar uma árvore de regressão que pudesse ajudar a prever renda dos novos futuros clientes. 
Ao utilizar técnicas de aprendizado de máquina, a instituição financeira visa oferecer uma experiência mais 
personalizada e eficiente, contribuindo para a satisfação e fidelização dos clientes.

## Bibliotecas Utilizadas 📚

```bash
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
```

## Desenvolvimento do Projeto 🛠️

O projeto segue um processo de análise de dados, incluindo etapas de limpeza e transformação, 
essenciais para assegurar o alcance do objetivo central: o treinamento de um modelo eficaz. Inicialmente, 
uma análise detalhada é conduzida utilizando a ferramenta "ProfileReport". Esta ferramenta gera um relatório 
abrangente, apresentando gráficos e insights iniciais sobre os dados. Após essa análise exploratória, a 
correlação entre os diferentes conjuntos de dados é examinado.

Na seção dedicada à "Previsão de Renda", o modelo treinado entra em ação. Aqui, é possível aplicar nossos próprios 
conjuntos de dados, permitindo que o modelo realize previsões imediatas de renda. Esse recurso proporciona uma visão 
valiosa e instantânea, contribuindo para a tomada de decisões informadas no contexto financeiro da instituição.

## Índice ✔️

```bash
- Etapa 1 Crisp-DM: Entendimento do Negócio
- Etapa 2 Crisp-DM: Entendimento dos Dados
- Etapa 3 Crisp-DM: Preparação dos Dados
- Etapa 4 Crisp-DM: Modelagem
- Etapa 5 Crisp-DM: Avaliação dos Resultados
- Etapa 6 Crisp-DM: Implantação
```
