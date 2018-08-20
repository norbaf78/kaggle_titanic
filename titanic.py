# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 19:58:00 2018

@author: Fabio Roncato
"""

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
# Helper libraries
#import numpy as np
#import matplotlib.pyplot as plt
#print(tf.__version__)

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
OUTPUT_PATH = 'data/submission.csv'

#read he training data with pandas
df_raw = pd.read_csv(TRAIN_PATH)
print(df_raw.info() )
df_raw.sample(5)


# PassengerId.  Unique identification of the passenger. It shouldn't be necessary for the machine learning model.
# Survived.     ->Survival (0 = No, 1 = Yes). Binary variable that will be our target variable.
# Pclass.       Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd). Ready to go.
# Name.         Name of the passenger.    We need to parse before using it.
# Sex.          Sex. Categorical variable that should be encoded.
# Age.          Age in years. Ready to go.
# SibSp.        # of siblings / spouses aboard the Titanic. Ready to go.
# Parch.        # of parents / children aboard the Titanic. Ready to go.
# Ticket.       Ticket number. Big mess. We need to understand its structure first.
# Fare.         Passenger fare. Ready to go.
# Cabin.        Cabin number. It needs to be parsed.
# Embarked.     Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). Categorical feature that should be encoded.


# create a copy
df = df_raw.copy()
# drop the information irrilevant ('PassengerId', 'Name') and the other more complicated ( 'Ticket', 'Cabin')
df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True)
#extract from the 'Name' the 'status' of the passengers and add this coulumn
df['Status'] = df_raw['Name'].str.extract('(\w+)\.', expand=False)
print(df['Status'].value_counts() )
#replace the status with 'Rare' for 'Status' with few entry
df['Status'] = df['Status'].str.replace(r'Dr|Rev|Col|Mlle|Major|Lady|Sir|Don|Capt|Mme|Jonkheer|Countess|Ms', 'Rare')
print(df['Status'].value_counts() )
#for each 'Status' we define the mean age to sostitute for the missing value for age
status_age_dict = df.groupby('Status')['Age'].mean().to_dict()
status_age_dict
#get the index for the entry with null value
no_age_index = df[df['Age'].isnull()].index
#sostitute the null value for the age with the mean in that category 'status'
df.loc[no_age_index, 'Age'] = df['Status'].loc[no_age_index].map(status_age_dict)
print(df.info())
#drop the entry where the 'Embarked' is null
df.drop(index=df.loc[df['Embarked'].isnull()].index, inplace=True)
print(df.info())


#----------------------------------------------------


def categorize_columns(df, colnames):
    for colname in colnames:
        df[colname] = df[colname].astype('category')
    return

categorize_columns(df, ['Pclass', 'Sex', 'Embarked', 'Status'])


df['Relatives'] = df['SibSp'] + df['Parch']
df.drop(columns=['SibSp', 'Parch'], inplace=True)



sns.barplot(df['Pclass'], df['Survived'])
plt.show()
sns.barplot(df['Sex'], df['Survived'])
plt.show()
sns.barplot(df['Embarked'], df['Survived'])
plt.show()
sns.barplot(df['Status'], df['Survived'])
plt.show()


