import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

gender_submission = pd.read_csv("gender_submission.csv")
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

print(gender_submission.head())
print(test_data.head())
print(train_data.head())

print(gender_submission.isnull().sum())
print(test_data.isnull().sum())
print(train_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

sns.countplot(x='Survived', data=train_data)
plt.title('Distribution of Survival')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title('Survival Based on Pclass')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('Survival Based on Sex')
plt.show()

sns.histplot(x='Age', hue='Survived', data=train_data, kde=True)
plt.title('Survival Based on Age')
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=train_data)
plt.title('Survival Based on Embarked')
plt.show()

numeric_columns = train_data.select_dtypes(include=[np.number]).columns
correlation_matrix = train_data[numeric_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
