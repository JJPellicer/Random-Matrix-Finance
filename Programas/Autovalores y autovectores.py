import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import os

# Lista de activos (nombre de archivos sin .csv)
assets = [
     'aex', 'cac40', 'dax', 'ftse100', 'ibex35', 'omxs30', 'smi',
    'bovespa', 'spmerval',
    'szse', 'shanghai', 'chinaa50',
    'hangseng', 'kospi', 'nikkei225', 'nifty50', 'spasx200',
    'dowjones','nasdaq', 'sp500', 'spbmvipc', 'sptsx',
    'oil', 'gas', 'gold', 'silver', 'copper', 'us10y'
]

# Ruta donde están tus archivos CSV
data_path = r'C:\Users\Juan\Documents\GitHub\Random-Matrix-Finance\Datos'

# Diccionario para guardar las series de precios
prices = {}

for asset in assets:
    filepath = os.path.join(data_path, f"{asset}.csv")

    # Leer el archivo con codificación española
    df = pd.read_csv(filepath, encoding='ISO-8859-1')
    df.columns = [col.lower().strip() for col in df.columns]  # normaliza cabeceras

    # Detectar la columna de fecha automáticamente
    date_col = next((col for col in df.columns if 'fecha' in col), None)
    if date_col is None:
        raise ValueError(f"No se encontró columna de fecha en {asset}.csv")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    # Usar la primera columna de datos ("Último")
    series_str = df.iloc[:, 0].astype(str).str.replace('.', '', regex=False).str.replace(',', '.')
    prices[asset] = series_str.astype(float)

# Combinar todas las series en un DataFrame
df_prices = pd.concat(prices, axis=1)

# Eliminar fechas con datos faltantes
df_prices = df_prices.dropna()


# Calcular rendimientos logarítmicos diarios
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()
print(df_returns.shape)
# Calcular matriz de correlación
correlation_matrix = df_returns.corr()

# Visualización del heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=False, cmap="RdBu_r", center=0,
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# AUTOVALORES Y AUTOVECTORES ORDENADOS
# ---------------------------------------------

eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]  # columnas = autovectores

#Plot histograma autovalores.
plt.figure(figsize=(10, 6))
plt.hist(eigenvals, bins=100, density=True, alpha=0.5, edgecolor='black', label='Eigenvalues')

plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Mostrar los 5 principales autovalores y sus componentes
for i in range(5):
    print(f"\n🔹 Autovalor {i+1}: {eigenvals[i]:.4f}")
    components = pd.Series(eigenvecs[:, i], index=correlation_matrix.columns)
    components_sorted = components.sort_values(ascending=False)
    print("Activos que más contribuyen:")
    print(components_sorted.head(28))

# ---------------------------------------------
# HEATMAP ORDENADO SEGÚN EL MARKET MODE
# ---------------------------------------------

# 1. Tomar el market mode (primer autovector)
market_mode = eigenvecs[:, 0]
market_mode_series = pd.Series(market_mode, index=correlation_matrix.columns)

# 2. Ordenar los activos por la magnitud de su componente en el market mode
order = market_mode_series.sort_values(ascending=False).index

# 3. Reordenar la matriz
correlation_matrix_ordered = correlation_matrix.reindex(index=order, columns=order)

# 4. Dibujar el heatmap ordenado
plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix_ordered, cmap='RdBu_r', center=0,
            xticklabels=correlation_matrix_ordered.columns,
            yticklabels=correlation_matrix_ordered.columns)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# CONSTRUCCIÓN DE CARTERAS CON LOS 3 PRINCIPALES AUTOVECTORES
# ---------------------------------------------

portfolio_cum_all = {}
for i in range(3):
    weights = eigenvecs[:, i]
    weights /= np.sum(np.abs(weights))
    print(f"pesos{i+1}:", weights)
    returns = df_returns @ weights
    portfolio_cum_all[f"Autovalor {i+1}"] = np.exp(returns.cumsum())

# Comparación con sp500
if 'sp500' in df_returns.columns:
    sp500_cum = np.exp(df_returns['sp500'].cumsum())
    portfolio_cum_all['sp500'] = sp500_cum

# Comparación con Ibex35
if 'ibex35' in df_returns.columns:
    ibex35_cum = np.exp(df_returns['ibex35'].cumsum())
    portfolio_cum_all['ibex35'] = ibex35_cum

# Plot
plt.figure(figsize=(10, 6))
for label, series in portfolio_cum_all.items():
    plt.plot(series, label=label)

plt.xlabel("Date")
plt.ylabel("Cumulative Growth Factor")
plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

print("Multiplicadores finales acumulados:")
for label, series in portfolio_cum_all.items():
    print(f"{label}: {series.iloc[-1]:.4f}")
