import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from scipy import stats
def carregarArquivos():
    df = pd.read_csv('Features.csv')
    # Substituir valores ausentes por espaços em branco para colunas de string
    string_columns = ['Name','Studio', 'Type', 'Tags', 'Release_season', 'staff', 'Related_anime', 'Related_Mange', 'Description']
    df[string_columns] = df[string_columns].fillna(' ')
    # Substituir valores ausentes por -1 para colunas numéricas
    numeric_columns = ['Episodes', 'Release_year', 'End_year']
    # Remover linhas com valores ausentes na coluna 'Rating'
    df = df.dropna(subset=['Rating'])
    df[numeric_columns] = df[numeric_columns].fillna(1)
    return df


    



def deteccaoAnomalias(df,feature1,feature2):
    # Selecionando as características relevantes para a detecção de anomalias
    features = [feature1,feature2]

    # Convertendo colunas relevantes para valores numéricos
    label_encoder = LabelEncoder()
    for feature in features:
        df[feature] = label_encoder.fit_transform(df[feature])

    # Criando o modelo de detecção de anomalias
    clf = IsolationForest(contamination=0.004, random_state=42)
    

    # Treinando o modelo
    clf.fit(df[features])

    # Obtendo as previsões de anomalia (-1 para anomalia, 1 para instância normal)
    y_pred = clf.predict(df[features])

    # Filtrando o DataFrame para manter apenas as instâncias consideradas normais
    df_normal = df[y_pred == 1]


    return df_normal


def analisePopularidade(df):
    # Selecionando as características e o alvo
    features = ['Name','Episodes', 'Studio', 'Release_year','Type','Tags','Release_season','staff', 'Related_anime', 'Related_Mange','cluster','Description',
                'quant_anime','quant_dub','quant_staff','quant_manga']
    target = 'Rating'

    # Substituir valores ausentes por espaços em branco

    # Convertendo a variável alvo em classes de popularidade

    df['Popularity'] = pd.cut(df[target], bins=[float('-inf'), 3.7, float('inf')], labels=['malAvaliado', 'bemAvaliado'])

    # Separando os conjuntos de treinamento e teste
    X = df[features]
    y = df['Popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Codificando variáveis categóricas
    categorical_features = ['Name','Studio','Type','Tags','Release_season','staff', 'Related_anime', 'Related_Mange','cluster','Episodes', 'Release_year','Description',
                             'quant_anime','quant_dub','quant_staff','quant_manga']
    categorical_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    

    X_train_encoded = categorical_transformer.fit_transform(X_train)
    X_test_encoded = categorical_transformer.transform(X_test)


    # Treinando o modelo 

    classifier = SVC(C=1.5, kernel='linear', class_weight={'malAvaliado': 1, 'bemAvaliado': 4},probability=True, random_state=42)



    classifier.fit( X_train_encoded, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = classifier.predict(X_test_encoded)

    # Avaliando o desempenho do modelo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}%'.format(precision * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('F1: {:.2f}%'.format(f1 * 100))
    print(classification_report(y_test, y_pred))



def correlacao(df):
    # Codificando variáveis categóricas
    categorical_features = ['Name','Studio','Type','Tags','Release_season','staff', 'Related_anime', 'Related_Mange','cluster','Episodes', 'Release_year','Description',
                             'quant_anime','quant_dub','quant_staff','quant_manga','Rating']
    categorical_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    
    # Aplicar a transformação nos dados
    encoded_data = categorical_transformer.fit_transform(df)

    # Calcular o coeficiente de correlação de ponto bisserial para cada variável
    correlations = {}
    for feature in categorical_features:
        if feature != 'Rating':
            r, p = stats.pointbiserialr(encoded_data[:, df.columns.get_loc(feature)], df['rating'])
            correlations[feature] = r

    # Ordenar as correlações em ordem decrescente
    correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Imprimir as correlações
    for feature, correlation in correlations:
        print(f'{feature}: {correlation}')


    

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







# obter o tempo atual antes de executar o código
start_time = time.time()

print('ETAPA 1 - Carregar arquivos de treinamento e teste...')
df = carregarArquivos()
print(df.columns)


print('ETAPA 2 - Associação de generos')
#df=AssociacaoGeneros(df)

print('ETAPA 3 - Deteccao de Anomalias')
df=deteccaoAnomalias(df,'Type','Episodes')
correlacao(df)

print('ETAPA 4 - Analise da popularidade')
analisePopularidade(df)

# obter o tempo atual depois de executar o código
end_time = time.time()

# calcular o tempo total de execução em segundos
total_time = end_time - start_time

print("Tempo de execução: {:.2f} segundos".format(total_time))