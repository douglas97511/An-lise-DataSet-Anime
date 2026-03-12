import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def carregarArquivos():
    df = pd.read_csv('Features.csv')
    
    # Substituir valores ausentes por espaços em branco para colunas de string
    string_columns = ['Studio', 'Type', 'Tags', 'Release_season', 'staff', 'Related_anime', 'Related_Mange', 'Description']
    df[string_columns] = df[string_columns].fillna(' ')
    # Substituir valores ausentes por -1 para colunas numéricas
    numeric_columns = ['Episodes', 'Release_year', 'End_year']
    # Remover linhas com valores ausentes na coluna 'Rating'
    df = df.dropna(subset=['Rating'])
    df[numeric_columns] = df[numeric_columns].fillna(1)
    return df

def analiseExploratoria(df):
    # Análise dos estúdios mais proeminentes
    estudio_count = df['Studio'].value_counts().head(20)  # Contagem de ocorrências dos estúdios
    print(estudio_count)

    # Gráfico de barras dos estúdios mais proeminentes
    plt.figure(figsize=(10, 6))
    estudio_count.plot(kind='bar')
    plt.title('Estúdios mais proeminentes')
    plt.xlabel('Estúdio')
    plt.ylabel('Contagem')
    plt.show()

    # Análise das tags mais comuns
    tags_count = df['Tags'].str.split(', ', expand=True).stack().value_counts().head(20)  # Contagem de ocorrências das tags
    print(tags_count)

    # Gráfico de barras das tags mais comuns
    plt.figure(figsize=(10, 4))
    tags_count.plot(kind='bar')
    plt.title('Tags mais comuns')
    plt.xlabel('Tag')
    plt.ylabel('Contagem')
    plt.show()


    # Contagem dos tipos de anime
    type_count = df['Type'].value_counts()

    # Plotar o gráfico de barras
    plt.figure(figsize=(10, 6))
    type_count.plot(kind='bar')
    plt.title('Tipos de anime')
    plt.xlabel('Tipo')
    plt.ylabel('Contagem')
    plt.show()


    # Calcular a quantidade média de episódios por tipo de anime
    average_episodes = df.groupby('Type')['Episodes'].mean()

    # Plotar o gráfico de barras
    plt.figure(figsize=(10, 6))
    average_episodes.plot(kind='bar')
    plt.title('Quantidade Média de Episódios por Tipo de Anime')
    plt.xlabel('Tipo de Anime')
    plt.ylabel('Quantidade Média de Episódios')
    plt.show()


def mapaCorrelacao(df): 
    # Selecionar as colunas relevantes
    columns = ['Episodes', 'Release_year',
            'quant_anime','quant_dub','quant_staff','quant_manga', 'Rating','cluster','Topic']

    # Filtrar o DataFrame com as colunas selecionadas
    data = df[columns]

    # Calcular a matriz de correlação
    corr_matrix = data.corr()

    # Plotar o mapa de calor da correlação
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Mapa de Calor de Correlação')
    plt.show()

def AssociacaoGeneros(df):
    
    # Pré-processamento dos dados
    generos_animes = df['Tags'].apply(lambda x: x.split(','))

    # Remover valores nulos ou vazios
    generos_animes = generos_animes.dropna()

    # Codificar as transações usando TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit_transform(generos_animes)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Executar o algoritmo FP-growth
    frequent_itemsets = fpgrowth(df_encoded, min_support=0.012, use_colnames=True)

    # Filtrar conjuntos frequentes com mais de 2 elementos
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] >=3]

    # Exibir os conjuntos frequentes
    print(frequent_itemsets)

    # Ordenar os conjuntos frequentes por frequência de ocorrência
    frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)


    # Plotar os conjuntos frequentes
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(frequent_itemsets)), frequent_itemsets['support'])
    plt.xticks(range(len(frequent_itemsets)), frequent_itemsets['itemsets'].apply(', '.join), rotation='vertical')
    plt.xlabel('Conjuntos Frequentes')
    plt.ylabel('Frequência de Ocorrência')
    plt.title('Conjuntos Frequentes')
    plt.tight_layout()
    plt.show()





# obter o tempo atual antes de executar o código
start_time = time.time()

print('ETAPA 1 - Carregar arquivos de treinamento e teste...')
df = carregarArquivos()
print(df.columns)


print('ETAPA 2 - Analise exploratoria do dataframe')
analiseExploratoria(df)
mapaCorrelacao(df)
#AssociacaoGeneros(df)

# obter o tempo atual depois de executar o código
end_time = time.time()

# calcular o tempo total de execução em segundos
total_time = end_time - start_time

print("Tempo de execução: {:.2f} segundos".format(total_time))