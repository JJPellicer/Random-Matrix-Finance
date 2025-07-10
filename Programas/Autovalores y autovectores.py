import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import os

# Lista de activos
assets = [
     'aex', 'cac40', 'dax', 'ftse100', 'ibex35', 'omxs30', 'smi',
    'bovespa', 'spmerval',
    'szse', 'shanghai', 'chinaa50',
    'hangseng', 'kospi', 'nikkei225', 'nifty50', 'spasx200',
    'dowjones','nasdaq', 'sp500', 'spbmvipc', 'sptsx',
    'oil', 'gas', 'gold', 'silver', 'copper', 'us10y'
]

# Ruta de archivos
data_path = r'C:\Users\Juan\Documents\GitHub\Random-Matrix-Finance\Datos'

# Diccionario precios
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

print(df_prices.head())


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
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# HEATMAP DE UNA MATRIZ DE CORRELACIÓN SHUFFLEADA (DE TODO EL PERIODO)
# ---------------------------------------------

# Barajar todo el dataframe de retornos por columnas
df_returns_shuffled = df_returns.copy()
for col in df_returns_shuffled.columns:
    df_returns_shuffled[col] = np.random.permutation(df_returns_shuffled[col].values)

# Calcular la matriz de correlación shuffleada global
corr_matrix_shuffled_full = df_returns_shuffled.corr()

# Plot Heat Map Shuffle
plt.figure(figsize=(8, 7))
sns.heatmap(corr_matrix_shuffled_full, cmap='RdBu_r', center=0,
            vmin=-0.1, vmax=0.1,
            xticklabels=corr_matrix_shuffled_full.columns,
            yticklabels=corr_matrix_shuffled_full.columns)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()


#Comprobación histograma matriz shuffle global
eigenvals_shuffled_full = np.linalg.eigvalsh(corr_matrix_shuffled_full)

# Histograma
plt.figure(figsize=(10, 6))
plt.hist(eigenvals_shuffled_full, bins=70, density=True, alpha=0.7, edgecolor='black')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.title("Distribución de autovalores (shuffle global de toda la matriz)")
plt.grid(True)
plt.tight_layout()
plt.show()


# ---------------------------------------------
# AUTOVALORES Y AUTOVECTORES ORDENADOS
# ---------------------------------------------

eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# Histohrama empírico autovalores correlation matrix.
plt.figure(figsize=(10, 6))
plt.hist(eigenvals, bins=100, density=True, alpha=0.5, edgecolor='black', label='Eigenvalues')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\rho(\lambda)$')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

#-------------------------
# Ley de Marchenko-Pastur
#-------------------------

#Calculo q Ley de Marchenko-Pastur
block_size = 114  # Tamaño de bloque deseado
n_blocks = df_returns.shape[0] // block_size
q_target = df_returns.shape[1] / block_size

print(f"Se usarán {n_blocks} bloques de {block_size} días. q = N/T = {q_target:.3f}")

# Calcular λ_min y λ_max teóricos para este q
lambda_min = (1 - np.sqrt(q_target))**2
lambda_max = (1 + np.sqrt(q_target))**2
lambda_vals = np.linspace(lambda_min * 0.8, lambda_max * 1.2, 500)
mp_density = (1 / (2 * np.pi * q_target * lambda_vals)) * np.sqrt(
    np.maximum(0, 4*lambda_vals*q_target-(lambda_vals+q_target-1)**2)
)
all_eigenvals = []

for i in range(n_blocks):
    # Extraer el bloque de datos
    block = df_returns.iloc[i * block_size:(i + 1) * block_size].copy()

    # Barajar columnas individualmente
    for col in block.columns:
        block[col] = np.random.permutation(block[col].values)

    # Calcular matriz de correlación y autovalores
    corr_block = block.corr()
    eigenvals_block = np.linalg.eigvalsh(corr_block)
    all_eigenvals.extend(eigenvals_block)

# Convertir a array de NumPy
all_eigenvals = np.array(all_eigenvals)

# Plot Marchenko-Pastur
plt.figure(figsize=(10, 6))
plt.hist(all_eigenvals, bins=100, density=True, alpha=0.5, color='orange', edgecolor='black', label='Shuffled blocks eigenvalues')
plt.plot(lambda_vals, mp_density, 'r--', label=f'Marchenko–Pastur (q ≈ {q_target:.3f})')
plt.axvline(lambda_min, color='black', linestyle='dotted', label=r'$\lambda_{-}$')
plt.axvline(lambda_max, color='black', linestyle='dotted', label=r'$\lambda_{+}$')
plt.xlabel(r'$\lambda$',fontsize=15)
plt.ylabel(r'$\rho(\lambda)$',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.tight_layout()
plt.show()

print("lamda max:",lambda_max,"lambda_min:",lambda_min)

#Plot comparativo Marchenko-Pastur vs Empírico.
plt.figure(figsize=(10, 6))
plt.hist(eigenvals, bins=100, range=(0,12.5), density=True, alpha=0.5, edgecolor='black', label='Empirical')
plt.hist(all_eigenvals, bins=100, range=(0,12.5), density=True, alpha=0.5, edgecolor='black', label='Shuffle')
plt.plot(lambda_vals, mp_density, 'r--', label=f'Marchenko–Pastur (q ≈ {q_target:.3f})')
plt.axvline(lambda_min, color='black', linestyle='dotted', label=r'$\lambda$_')
plt.axvline(lambda_max, color='black', linestyle='dotted', label=r'$\lambda_{\plus}$')
plt.xlabel(r'$\lambda$',fontsize=15)
plt.ylabel(r'$\rho(\lambda)$',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
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
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
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
    portfolio_cum_all[f"Eigenvalue {i+1}"] = np.exp(returns.cumsum())

# Comparación con sp500
if 'sp500' in df_returns.columns:
    sp500_cum = np.exp(df_returns['sp500'].cumsum())
    portfolio_cum_all['sp500'] = sp500_cum

# Comparación con Ibex35
if 'ibex35' in df_returns.columns:
    ibex35_cum = np.exp(df_returns['ibex35'].cumsum())
    portfolio_cum_all['ibex35'] = ibex35_cum

# Plot rendimientos acumulados
plt.figure(figsize=(10, 6))
for label, series in portfolio_cum_all.items():
    plt.plot(series, label=label)

plt.xlabel("Date",fontsize=15)
plt.ylabel("Cumulative Growth Factor",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.show()

#Rendimientos obtenidos
print("Multiplicadores finales acumulados:")
for label, series in portfolio_cum_all.items():
    print(f"{label}: {series.iloc[-1]:.4f}")
print("N autovalores reales:", len(eigenvals))
print("N autovalores shuffle:", len(all_eigenvals))