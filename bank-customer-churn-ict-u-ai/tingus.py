'''
About Dataset:
RowNumber—corresponds to the record (row) number and has no effect on the output.
CustomerId—contains random values and has no effect on customer leaving the bank.
Surname—the surname of a customer has no impact on their decision to leave the bank.
CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
Geography—a customer’s location can affect their decision to leave the bank.
Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
NumOfProducts—refers to the number of products that a customer has purchased through the bank.
HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
IsActiveMember—active customers are less likely to leave the bank.
EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
Exited—whether or not the customer left the bank.
Complain—customer has complaint or not.
Satisfaction Score—Score provided by the customer for their complaint resolution.
Card Type—type of card hold by the customer.
Points Earned—the points earned by the customer for using credit card.
Acknowledgements
As we know, it is much more expensive to sign in a new client than keeping an existing one.
It is advantageous for banks to know what leads a client towards the decision to leave the company.
Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible.

'''
# pasos 1, importar as bibliotcas e os dados

import pandas as pd
import numpy as np
import catboost as cb
import plotly.express as px
import os

trainPath = r'C:\Users\Rafael\OneDrive - UNIOESTE\Documentos\Estudo Analise de dados - Keggle\bank-customer-churn-ict-u-ai\train.csv'

tabelaTreino = pd.read_csv(trainPath)

tabelaTreino.shape

tabelaTreino.info()

# passos 2, tratar os dados

tabelaTreino = tabelaTreino.drop(columns=['id','CustomerId', 'Surname'])

tabelaTreino.info()



Categoricos = ['Geography', 
               'Gender',
               'Tenure',
               'NumOfProducts',
               'HasCrCard',
               'IsActiveMember',]

# paso 3, analise de dados iniciais
'''
# paso 3, analise de dados iniciais
import plotly.express as px

for coluna in tabelaTreino.columns:
    if coluna != 'Exited':  # Pular a coluna Exited
        grafico = px.histogram(tabelaTreino, x=coluna, color='Exited', title=coluna)
        grafico.show()

        # Criar tabela cruzada normalizada por linha (100% por categoria)
        dados = pd.crosstab(tabelaTreino[coluna], tabelaTreino['Exited'], normalize='index') * 100
        dados = dados.reset_index().melt(id_vars=coluna)
        dados.columns = [coluna, 'Exited', 'Percentage']
        
        # Criar gráfico com porcentagens
        grafico = px.bar(dados, x=coluna, y='Percentage', color='Exited', 
                        barmode='group', title=coluna,
                        labels={'Percentage': 'Porcentagem (%)', 'Exited': 'Cancelou'},
                        text='Percentage')
        grafico.update_traces(textposition='auto', texttemplate='%{text:.1f}%')

        grafico.show()

        grafico = px.line(dados, x=coluna, y='Percentage', color='Exited',title=coluna)
        grafico.show()
'''

'''
# Criar uma tabela cruzada normalizada por linha (100% por categoria) para descobrir a correlação entre as variaveis
# com o objetivo de descobrir quais variáveis tem mais correlação com a variável alvo (Exited) e quais variáveis tem mais correlação entre si, para depois fazer um feature selection, ou seja, selecionar as variáveis mais importantes para o modelo

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

# só não aplicamos na coluna score_credito que é o nosso objetivo (y || f(x)), ou seja, a variável que queremos prever
for coluna in tabelaTreino.columns:
    if tabelaTreino[coluna].dtype == 'str' and coluna != 'Exited':
        tabelaTreino[coluna] = codificador.fit_transform(tabelaTreino[coluna])


import seaborn as sns

sns.heatmap(tabelaTreino.corr(), annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt = '.2f', )
'''

from sklearn.model_selection import train_test_split

# Separando as variáveis independentes de y
X = tabelaTreino.drop(columns=['Exited'])  # Variáveis independentes (todas as colunas, exceto 'Exited')

# Separando a variável dependente (y - O diagnóstico real)
y = tabelaTreino['Exited']

# Verificando as separações
print("Formato do X (Variáveis):", X.shape)
print("Formato do y (Target):", y.shape)

# Calcula a porcentagem de cada classe dentro de y
proporcoes = y.value_counts(normalize=True) * 100

print("Proporção das classes no dataset:")
print(proporcoes)

x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)

print("Pacientes para o modelo estudar (X_train):", x_treino.shape)
print("Pacientes para o modelo testar (X_test):", x_teste.shape)


#antes de treinar o modelo, precisamos fazer um feature selection, ou seja, selecionar as variáveis mais importantes para o modelo
