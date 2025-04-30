# Cálculo y visualización de rendimientos logarítmicos estandarizados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import gaussian_kde, norm

# Lista de activos
assets = [
    'aex', 'bovespa', 'cac40', 'chinaa50', 'copper', 'dax', 'dowjones', 'ftse100',
    'gas', 'gold', 'hangseng', 'ibex35', 'kospi', 'nasdaq', 'nifty50',
    'nikkei225', 'oil', 'omxs30', 'shanghai', 'silver', 'smi', 'sp500', 'spasx200',
    'spbmvipc', 'spmerval', 'sptsx', 'szse', 'us10y'
]

# Ruta a los datos
data_path = 'C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos'

# Diccionario de precios
prices = {}

for asset in assets:
    filepath = os.path.join(data_path, f"{asset}.csv")
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df.columns = [col.lower().strip() for col in df.columns]
    date_col = next((col for col in df.columns if 'fecha' in col), None)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    series_str = df.iloc[:, 0].astype(str).str.replace('.', '', regex=False).str.replace(',', '.')
    prices[asset] = series_str.astype(float)

# Unir todos los precios y calcular rendimientos
df_prices = pd.concat(prices, axis=1).dropna()
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# Estandarización de rendimientos
std_returns = (df_returns - df_returns.mean()) / df_returns.std()

# Visualizar todas las curvas en una única gráfica
plt.figure(figsize=(10, 6))
x = np.linspace(-5, 5, 1000)
y_gauss = norm.pdf(x, 0, 1)
plt.plot(x, y_gauss, 'k--', label='Normal(0,1)', linewidth=2)

for asset in assets:
    sns.kdeplot(std_returns[asset], linewidth=2)

plt.title("Empirical distributions vs gaussian")
plt.xlabel("Standard Returns")
plt.ylabel("PDF")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

#Escala logaritmica

plt.figure(figsize=(10, 6))
x = np.linspace(-5, 5, 1000)

# Añadir curva gaussiana estándar
plt.plot(x, norm.pdf(x), 'k--', linewidth=2, label='Normal(0,1)')

# KDE de cada activo con filtro para valores pequeños
for asset in assets:
    data = std_returns[asset].dropna()
    kde = gaussian_kde(data, bw_method='scott')
    y_vals = kde(x)
    y_vals = np.clip(y_vals, 1e-6, None)  # Evitar valores cercanos a cero o negativos
    plt.plot(x, y_vals, linewidth=1)

plt.yscale('log')
plt.title("Empirical KDEs vs Gaussian (log-scale)")
plt.xlabel("Standard Returns")
plt.ylabel("PDF")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()