from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from textblob import TextBlob
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


# Carregar stopwords personal.
with open('StopWords.txt', 'r') as file:
    STOPWORDSPERSO =file.read().splitlines()

STOPWORDSPERSO = set(STOPWORDSPERSO)

STOPWORDS = stopwords.words('english')
PONTUACAO = string.punctuation
stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def remove_pontuacao(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word.lower() not in STOPWORDS])

def remove_numeros(text):
    return re.sub(r'\d+', '', text)

def remove_caracteres_especiais(text):
    return re.sub(r'[^\w\s]', '', text)



def preparaDF( df):

    string_columns = ['Studio', 'Type', 'Tags', 'Release_season', 'staff', 'Related_anime', 'Related_Mange', 'Description','Content_Warning']
    df[string_columns] = df[string_columns].fillna(' ')
    # Substituir valores ausentes por -1 para colunas numéricas
    numeric_columns = ['Episodes', 'Release_year', 'End_year']
    df[numeric_columns] = df[numeric_columns].fillna(-1)
    # Remover linhas com valores ausentes na coluna 'Rating'
    df = df.dropna(subset=['Rating'])
    df['quant_anime'] = df.Related_anime.apply(lambda x: len(x.split(',')) if not pd.isna(x) else 0)
    df['quant_manga'] = df.Related_Mange.apply(lambda x: len(x.split(',')) if not pd.isna(x) else 0)
    df['quant_dub'] = df.Voice_actors.apply(lambda x: len(x.split(',')) if not pd.isna(x) else 0)
    df['quant_staff'] = df.staff.apply(lambda x: len(x.split(',')) if not pd.isna(x) else 0)
    return df



def preProcessamentoTexto(df):
    print('pré-processamento descricao')

    # Convertendo o texto para letras minúsculas
    df['Description'] = df['Description'].str.lower()

    # Remover pontuação dos textos
    df['Description'] = df['Description'].apply(lambda text: remove_pontuacao(text))

     # Remover números
    df['Description'] = df['Description'].apply(lambda text: remove_numeros(text))

    # Remover stopwords
    df['Description']=df['Description'].apply(lambda text: remove_stopwords(text))


    return df


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
    frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 3]


    # Aplicar clusterização aos dados
    num_clusters = len(frequent_itemsets)  # Usar o número de conjuntos frequentes como número de clusters
    kmeans = KMeans(n_clusters=num_clusters,n_init=10,random_state=42)
    kmeans.fit(df_encoded)

    # Obter as labels de cluster para cada instância
    labels = kmeans.labels_

    # Adicionar as labels de cluster como uma nova coluna no DataFrame original
    df['cluster'] = labels

    return df


def topicosDescricao(df):
    # Vetorização dos textos
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Description'].values.astype('U'))

    # Definição do modelo de análise de tópicos
    num_topics = 20  # Número de tópicos a serem identificados
    model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    # Treinamento do modelo
    model.fit(X)

    # Obtendo os tópicos identificados
    topic_labels = []  # Lista vazia para armazenar os rótulos dos tópicos
    topic_words = []  # Lista vazia para armazenar as palavras dos tópicos
    feature_names = vectorizer.get_feature_names_out()  # Obter os nomes das features (palavras)

    for topic_idx, topic in enumerate(model.components_):
        topic_words.append([feature_names[i] for i in topic.argsort()])  # Armazenar as palavras mais relevantes

    for i, row in df.iterrows():
        if row['Description'].isspace():
            topic_labels.append(-1)  # Atribuir valor -1 para descrição inválida
        else:
            topic_probs = model.transform(vectorizer.transform([row['Description']]))
            topic_label = int(topic_probs.argmax())  # Atribuir valor inteiro para tópico identificado
            topic_labels.append(topic_label)

    # Salvar os tópicos no DataFrame
    df['Topic'] = topic_labels
    return df




def salvarCSV(df, filename):
    # Salvar os sentimentos em um arquivo CSV
    df[[ 'Rank', 'Name', 'Japanese_name', 'Type', 'Episodes', 'Studio',
       'Release_season', 'Tags', 'Rating', 'Release_year', 'End_year',
       'Description', 'Content_Warning', 'Related_Mange', 'Related_anime',
       'Voice_actors', 'staff','quant_anime','quant_dub','cluster','quant_staff','quant_manga','Topic']].to_csv(filename, index=False)
    




# Carregar arquivos
df= pd.read_csv('anime.csv')
df= preparaDF(df)
df= preProcessamentoTexto(df)
df=AssociacaoGeneros(df)
df=topicosDescricao(df)
salvarCSV(df, 'Features.csv')