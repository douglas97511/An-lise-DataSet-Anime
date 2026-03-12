
Anime Rating Analysis and Classification

Este projeto foi desenvolvido para a disciplina de Mineração de Dados, com o objetivo de analisar quais características estão associadas a animes bem avaliados e identificar padrões nos animes preferidos pelos usuários.

Utilizando um dataset público com informações sobre milhares de animes, realizamos uma análise exploratória de dados, engenharia de features e aplicamos técnicas de Machine Learning e mineração de dados para classificar animes entre bem avaliados e mal avaliados.

Projeto desenvolvido em colaboração com João Victor Schamne.

O dataset utilizado foi o Anime Dataset 2022, disponível no Kaggle:

Dataset:
https://www.kaggle.com/datasets/vishalmane10/anime-dataset-2022

Fonte original dos dados:
https://www.anime-planet.com/

  Etapas do Projeto
  
1.Análise Exploratória de Dados (EDA)

-Estúdios mais frequentes

-Tipos de mídia mais comuns

-Gêneros mais populares

-Relações entre diferentes features

-Distribuição das avaliações

Também foram exploradas combinações frequentes de tags e gêneros.

2. Pré-processamento e Engenharia de Features

-Tratamento de valores ausentes

-Conversão e seleção de features relevantes

-Processamento de texto das descrições dos animes

-Processamento de Texto

-As descrições foram tratadas com:

-Conversão para minúsculas

-Remoção de pontuação

-Remoção de números

-Remoção de stopwords

Em seguida foi aplicada a técnica CountVectorizer para transformar o texto em vetores numéricos.

3. Modelagem de Tópicos

Foi aplicado Latent Dirichlet Allocation (LDA) para identificar os principais tópicos presentes nas descrições dos animes.

Isso permitiu extrair temas predominantes e utilizar essas informações como novas features para análise.

4.Mineração de Padrões

-FP-Growth para encontrar combinações frequentes de tags

-K-Means para clusterização dos animes com base nessas combinações

5. Detecção de Anomalias

Para remover dados inconsistentes, foi aplicado o algoritmo:

Isolation Forest

Esse método permitiu identificar e remover registros considerados anômalos antes da etapa de classificação.

6. Classificação

O objetivo do modelo foi classificar animes entre:

-Bem avaliados

-Mal avaliados

A classificação foi baseada na feature rating:

-Rating > 3.7 → Bem avaliado

-Rating ≤ 3.7 → Mal avaliado

Foi utilizado o algoritmo:

Support Vector Classifier (SVC)

7.Resultados

O modelo de classificação apresentou os seguintes resultados:

Métrica	Resultado
Accuracy	90.62%
Precision	90.13%
Recall	90.62%
F1-Score	90.06%

Observações:

-A classe mal avaliado apresentou melhor desempenho

-Houve impacto do desbalanceamento das classes

-Algumas features mostraram baixa correlação com o rating


8. Insights

Alguns resultados interessantes observados:

-Certas combinações de gêneros aparecem com maior frequência

-A quantidade de staff e dubladores pode influenciar na avaliação

-Animes com determinados tipos de produção apresentam padrões de popularidade

-O dataset apresenta desbalanceamento de classes, o que influencia a performance dos modelos

9. Tecnologias Utilizadas

Python

Pandas

NumPy

Scikit-learn

NLP (CountVectorizer)

Latent Dirichlet Allocation (LDA)

FP-Growth

K-Means

Isolation Forest

Support Vector Machine (SVM)
