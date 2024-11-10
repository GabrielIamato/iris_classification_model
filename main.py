import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
## Label Encoder para tratar variáveis categóricas
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def apresentando_metricas(test_labels, y_pred):
    ## Calculando métricas
    acc = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, average = 'weighted')
    recall = recall_score(test_labels, y_pred, average = 'weighted')
    f1 = f1_score(test_labels, y_pred, average = 'weighted')
    ## Exibindo métricas
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
def gera_matriz_confusao(test_labels_aux, y_pred_aux):
    ## gerando matriz de confusão
    cm = confusion_matrix(test_labels_aux, y_pred_aux, labels = [0,1,2])
    
    ## plotando matriz de confusão com seaborn
    # plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot = True, fmt='d', cmap = 'Blues',  xticklabels=['setosa', 'versicolo', 'virginica'], 
                yticklabels=['setosa', 'versicolo', 'virginica'], cbar=True)
    
    # Adicionando título e labels
    plt.title('Matriz de Confusão')
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    
    # Mostrar o gráfico
    plt.show()
class Modelo():
    def __init__(self):
        pass

    def CarregarDataset(self, path):
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        self.df = pd.read_csv(path, names=names)
    def VisualizarDataset(self, linhas=5):
        print(self.df.head(linhas))
    def TratamentoDeDados(self):
        ## Tratamento de variável categórica
        label_encoder = LabelEncoder()
        especies = label_encoder.fit_transform(self.df['Species'])

        ## Normalização por MinMaxScaler
        features = self.df.drop('Species', axis=1)
        scaler = preprocessing.MinMaxScaler()
        df_normalizado = scaler.fit_transform(features)
        names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.df = pd.DataFrame(df_normalizado, columns = names)
        ## Transformar em int facilita o modelo a entender que são CLASSES, mesmo que os dados estejam formatados em: 0.0, 1.0 e 2.0
        self.df['Species'] = especies
        self.df['Species'] = self.df['Species'].astype(int)
        # print("Classes codificadas:", self.df['Species'].unique())

    def Treinamento(self, modelo):
        # print(self.df)
        ### Variável target
        labels = np.array(self.df['Species'])
        ### Variáveis de treinamento do modelo
        features = self.df.drop('Species', axis = 1)
        
        feature_list = list(features.columns)
        
        features = np.array(features)

        ## Separando em treino e teste
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, 
                                                                                    test_size = 0.25, random_state = 42)

         # Salvando os dados de teste como atributos da classe (para que sejam utilizados em teste de modelo)
        self.test_features = test_features
        self.test_labels = test_labels
        
        if(modelo == 0): ## Random Forest
            ## número de árvores
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            
            ## número de features a ser considerado em cada separação
            max_features = ['sqrt', 'log2', None]
            
            ## máximo número de levels em cada árvore
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            
            ## número mínimo de amostras para divisão do nó
            min_samples_split = [2, 5, 10]
            ## número mínimo de amostras em cada nó folha
            min_samples_leaf = [1, 2, 4]
            
            ## Método de selecionar amostras para treinar cada árvore
            bootstrap = [True, False]
            
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            
            rf = RandomForestClassifier()
            
            ## n_iter = número de iterações para busca de melhores hiperparâmetros
            ## cv = número de folds para cross-validation
            ## verbose = controls
            ## n_jobs = deixar mais rápido o processamento
            
            model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,cv = 10,
                                           verbose = 2, random_state = 42, n_jobs =-1)
            ## Depois de achados os melhores hiperparametros (ainda que não o máximo global),alimenta-se o modelo de novo
            ## Depois de "rf_random.fit", o modelo está treinado com cross-validation e com hyperparametros tunados
            # rf_random.fit(train_features, train_labels)
            # self.modelo = rf_random.best_estimator_

        if(modelo == 1): ##SVM
            ## Criando modelo
            model = SVC(random_state = 42)
            ## Selecionando hiperparâmetros teste para construir o modelo
            random_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
            ## Fazendo RandomizeSearch para procurar melhores parâmetros
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid, 
                                          n_iter = 10, scoring = 'accuracy', cv =3, random_state = 42, n_jobs = -1)


        if(modelo == 2): ## Naive Bayes
            ## Criando modelo
            model = GaussianNB()

            random_grid = {
                ## Suavização da variância
                'var_smoothing': np.logspace(0, -9, num=100), 
                ## Probabilidade inicial para as classes ([0,1,2])
                'priors': [None, [0.33, 0.33, 0.34], [0.5, 0.25, 0.25], [0.7, 0.2, 0.1]] 
            }
            
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                                          n_iter = 10, scoring = 'accuracy', cv=10, random_state = 42, n_jobs = -1)
        if(modelo == 3): ## K-Neighbors
            model = KNeighborsClassifier()
            random_grid = {
                'n_neighbors': np.arange(1, 21),  # Número de vizinhos considerados
                'weights': ['uniform', 'distance'],  # Função dos pesos: uniforme ou ponderada em distâncias
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Técnica de seleção de vizinhos
                'leaf_size': np.arange(20, 50),  # Para algoritmos baseados em árvore, tamanho das folhas
                'p': [1, 2],  # Cálculo de distância: 1 para Manhattan, 2 para Euclidiana
                  # Parâmetro de distância: 1 para Manhattan, 2 para Euclidiana
            }
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                                          n_iter = 10, scoring = 'accuracy', cv=10, random_state = 42, n_jobs = -1)
        if(modelo == 4): ##XGBoost (Extreme Gradient Boosting)
            model = XGBClassifier(eval_metric = 'mlogloss', random_state = 42)

            random_grid = {
                'n_estimators': [100, 200, 300, 400, 500],  # Quantidade de árvores
                'max_depth': [3, 4, 5, 6, 7],  # Profundidade máxima das árvores
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  # Taxa de aprendizado
                'subsample': [0.7, 0.8, 0.9, 1.0],  # Fração de amostras para cada ávore
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # Fração de características por árvore
                'gamma': [0, 0.1, 0.2, 0.3, 0.4],  # Regularização de crescimento da Árvore
                'reg_alpha': [0, 0.1, 0.2, 0.3],  # Regularização L1 (Lasso)
                'reg_lambda': [0, 0.1, 0.2, 0.3]  # Regularização L2 (Ridge)
            }
            model = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                                          n_iter = 10, scoring = 'accuracy', cv=10, random_state = 42, n_jobs = -1)
    
        ## Alimentando modelo treinado
        model.fit(train_features,train_labels)
        ## Guardando melhores parâmetros e hiperparâmetros
        self.modelo = model.best_estimator_
        print("Modelo treinado!!")

    def Teste(self):
        y_pred = self.modelo.predict(self.test_features)
        # rf_random.predict(test_features)
        apresentando_metricas(self.test_labels, y_pred)
        gera_matriz_confusao(self.test_labels, y_pred)
        
    def Train(self, modelo):
        self.CarregarDataset("iris.data")  # Carrega o dataset especificado.

        # Tratamento de dados opcional, pode ser comentado se não for necessário
        self.TratamentoDeDados()

        self.Treinamento(modelo)  # Executa o treinamento do modelo

        self.Teste()

