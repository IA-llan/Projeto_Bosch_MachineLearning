# Importando as bibliotecas que serao utilizadas

import pandas as pd
import numpy as np

# Importando o dataset que sera utilizado

dataset = pd.read_csv('C:/Users/allan/OneDrive/Área de Trabalho/Controles_das_Maquinas/SimulacaoSensores.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Dividindo o dataset entre variaveis para treino e para teste

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Dimensionando os recursos (deixando todos na mesma escala)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Treinando o modelo que será utilizado

from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

# Criando uma variável para guardar os valores previstos

y_pred = classifier.predict(x_test)

# Imprimindo uma comparação entre os valores do dataset e os previstos
print('Nesta tabela está sendo mostrado do lado esquerdo a previsao feita pelo modelo e do lado direito o real valor:')
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Imprimindo uma matriz que mostras quais foram os acertos e erros do modelo
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print('Nesta matriz sao mostradas os acertos e erros do modelo: ')
print(cm)
acuracia = accuracy_score(y_test, y_pred)
print('A precisão do modelo é de: ' + str(acuracia * 100) + '%')

# Prevendo um valor aleatório em que o usuário fornece os valores
valor_1 = int(input('Para testar o modelo, digite um valor para o primeiro sensor: '))
valor_2 = int(input('Para testar o modelo, digite um valor para o segundo sensor: '))
print(classifier.predict(sc.transform([[valor_1, valor_2]])))

previsao = classifier.predict(sc.transform([[valor_1, valor_2]]))

if previsao == 1:
    print('De acordo com os dados digitidados anteriormente a maquina está ok!')
else:
    print('A máquina pode estar apresentando algum problema!!')
