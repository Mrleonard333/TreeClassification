import pandas as pd
from os import system
from time import sleep
from sklearn.metrics import accuracy_score # < Precision
from sklearn.preprocessing import LabelEncoder # < Adapt
from sklearn.tree import DecisionTreeClassifier # < Module

        # v Will import the arquive
ARQ = pd.read_csv('Arquive.csv')
X = ARQ.drop('SM100k', axis=1) # < Input
Y = ARQ.SM100k # < Output

        # v Will turn the string into number for the training
LB = LabelEncoder()
DTC = DecisionTreeClassifier()
        # ^ Will import the module for the classification

# v Will adapt the graphic for the training v
X['L_company'] = LB.fit_transform(X['company'])
X['L_job'] = LB.fit_transform(X['job'])
X['L_degree'] = LB.fit_transform(X['degree'])
X = X.drop(['company', 'job', 'degree'], axis=1)
DTC.fit(X, Y)
# ^ Will train the machine

print('[YOUR SALARY IS BIGGER THAN 100.000?]\n')
sleep(0.5)
print('[0] Abc Pharma')
sleep(0.5)
print('[1] Facebook')
sleep(0.5)
print('[2] Google\n')
sleep(1)
COMPANY = int(input('[NUMBER] - Which of these companies you work for?: '))

system('cls')

print('[YOUR SALARY IS BIGGER THAN 100.000?]\n')
sleep(0.5)
print('[0] Business Manager')
sleep(0.5)
print('[1] Computer Programmer')
sleep(0.5)
print('[2] Sales Executive\n')
sleep(1)
JOB = int(input('[NUMBER] - Which of these jobs you do?: '))

system('cls')

print('[YOUR SALARY IS BIGGER THAN 100.000?]\n')
sleep(0.5)
print('[0] Bachelor')
sleep(0.5)
print('[1] Master\n')
sleep(1)
DEGREE = int(input('[NUMBER] - Which one are your level of knowledge?: '))

RESPONSE = DTC.predict([[COMPANY,JOB,DEGREE]])
                # ^ Will classificate the values
system('cls')

print('[YOUR SALARY IS BIGGER THAN 100.000?]\n')
sleep(0.5)                                              # v Will show the precision of the classification
print(f'Response[{RESPONSE[0]}] - Precision[{accuracy_score(Y, DTC.predict(X))}]')