import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('FuelConsumption.csv')

sns.set_style('whitegrid')

df.head()

# Check null values
df.isnull().sum()

# Check total null values
df.isnull().sum().sum()

# Number of rows and columns
df.shape

# Dataset information
df.info()

# Statistical summary
df.describe()

# Check column names
df.columns

# Check unique fuel types
df['FUELTYPE'].unique()

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x='FUELCONSUMPTION_COMB_MPG',
    y='CO2EMISSIONS'
)

plt.title('CO2 Emissions vs Combined MPG')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions')

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x='FUELCONSUMPTION_COMB_MPG',
    y='CO2EMISSIONS',
    hue='FUELTYPE',
    palette='Set1',
    alpha=0.7
)

plt.title('CO2 Emissions vs MPG by Fuel Type')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions (g/km)')

plt.legend(title='Fuel Type')
plt.show()

plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x='FUELCONSUMPTION_COMB_MPG',
    y='CO2EMISSIONS',
    hue='FUELTYPE',
    size='ENGINESIZE',
    sizes=(25,250),
    palette='Dark2',
    alpha=0.7
)

plt.title('CO2 Emissions vs MPG by Fuel Type and Engine Size')
plt.xlabel('Combined MPG')
plt.ylabel('CO2 Emissions (g/km)')

plt.legend(
    bbox_to_anchor=(1.05,1),
    loc='upper left'
)

plt.show()

correlation = df.corr(numeric_only=True)

print(correlation)

plt.figure(figsize=(12,9))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    center=0
)

plt.title('Correlation Between All Numerical Features')
plt.show()