import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# loading the dataset
df = pd.read_csv("C:\\Users\\dell\\Downloads\\titanic3.csv")
print(df)
print(df.info())
print(df.head())
# Drop columns with too many missing values
df = df.drop(columns=['cabin', 'boat', 'body', 'home.dest'], errors='ignore')

# Fill missing 'age' with mean
df['age'] = df['age'].fillna(df['age'].mean())

# Fill missing 'embarked' with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop rows with any remaining null values
df = df.dropna()

# Survival count plot
sns.countplot(data=df, x='survived')
plt.title('Survival Count')
plt.show()

# Survival by class by count plot
sns.countplot(data=df, x='pclass', hue='survived')
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution by histplot
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Survival by gender by count plot
sns.countplot(data=df, x='sex', hue='survived')
plt.title('Survival by Gender')
plt.show()

#scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='survived', palette='Set1')
plt.title('Fare vs Age (colored by Survival)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(True)
plt.show()

#box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='pclass', y='age', hue='pclass', palette='pastel', legend=False)
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.grid(True)
plt.show()

#violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='sex', y='age', hue='survived', split=True, palette='Set2')
plt.title('Age Distribution by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(True)
plt.show()

#kde plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['survived'] == 0], x='age', label='Did Not Survive', fill=True, color='red', alpha=0.5)
sns.kdeplot(data=df[df['survived'] == 1], x='age', label='Survived', fill=True, color='green', alpha=0.5)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

