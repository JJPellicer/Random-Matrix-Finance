import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Lista de activos (nombre de archivos sin .csv)
assets = [
    'aex', 'cac40', 'dax', 'ftse100', 'ibex35', 'omxs30', 'smi',
    'bovespa', 'spmerval',
    'chinaa50', 'shanghai', 'szse',
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

# Calcular matriz de correlación
correlation_matrix = df_returns.corr()

# Visualización del heatmap
plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0,
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns)
plt.title("Correlation matrix of logarithmic reutrns")
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

plt.title("Eigenvalues distribution")
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
    components_sorted = components.abs().sort_values(ascending=False)
    print("Activos que más contribuyen:")
    print(components_sorted.head(5))

# ---------------------------------------------
# CONSTRUCCIÓN DE CARTERAS CON LOS 5 PRINCIPALES AUTOVECTORES
# ---------------------------------------------

portfolio_cum_all = {}
for i in range(5):
    weights = eigenvecs[:, i]
    weights /= np.sum(np.abs(weights))
    returns = df_returns @ weights
    portfolio_cum_all[f"Autovalor {i+1}"] = (1 + returns).cumprod()

# Comparación con SP500 (si existe en los datos)
if 'sp500' in df_returns.columns:
    sp500_cum = (1 + df_returns['sp500']).cumprod()
    portfolio_cum_all['SP500'] = sp500_cum

# Plot
plt.figure(figsize=(12, 6))
for label, series in portfolio_cum_all.items():
    plt.plot(series, label=label)

plt.title("Cumulative Performance of Portfolios Based on Leading Eigenvectors")
plt.xlabel("Date")
plt.ylabel("Portfolios Multiplier")
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# MARCENKO-PASTUR TEÓRICA - FIGURA AJUSTADA
# ---------------------------------------------

# q1 = 3.45
# q2 = 2.0

# def rho_empirical(lambda_vals, q):
#     lambda_plus = (np.sqrt(q) + 1)**2
#     lambda_minus = (np.sqrt(q) - 1)**2
#     rho = np.zeros_like(lambda_vals)
#     mask = (lambda_vals >= lambda_minus) & (lambda_vals <= lambda_plus)
#     rho[mask] = np.sqrt(4 * lambda_vals[mask] * q - (lambda_vals[mask] + q - 1)**2) / (2 * np.pi * lambda_vals[mask] * q)
#     rho[~np.isfinite(rho)] = 0
#     return rho

# lambda_vals = np.linspace(0.01, 3, 1000)
# rho_q1 = rho_empirical(lambda_vals, q1)
# rho_q2 = rho_empirical(lambda_vals, q2)

# plt.figure(figsize=(8, 5))
# plt.plot(lambda_vals, rho_q2, label='exp Q=2', color='black')
# plt.plot(lambda_vals, rho_q1, '--', label='std Q=3.45', color='gray')
# plt.title('Marčenko–Pastur: densidad teórica (forma del paper)')
# plt.xlabel(r'$\lambda$')
# plt.ylabel(r'$\rho(\lambda)$')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
