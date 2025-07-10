import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import seaborn as sns
import os
from scipy.stats import gaussian_kde, norm, linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

# Crear la figura principal
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar la gaussiana teórica
x = np.linspace(-5, 5, 1000)
y_gauss = norm.pdf(x, 0,1)
ax.plot(x, y_gauss, 'k--', label='Normal(0,1)', linewidth=2)

# Histograma de cada activo
for asset in df_returns.columns:
    ax.hist(std_returns[asset], bins=100, density=True, alpha=0.3, histtype='stepfilled')

ax.set_xlabel("Standard Returns",fontsize=15)
ax.set_ylabel("PDF",fontsize=15)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.set_yscale('log')
ax.set_ylim(1e-3, 1)

# Inset: Cola izquierda
axins_left = inset_axes(ax, width="100%", height="100%",
                        bbox_to_anchor=(0.1, 0.6, 0.3, 0.3),
                        bbox_transform=ax.transAxes,
                        borderpad=0)

for asset in df_returns.columns:
    axins_left.hist(-std_returns[asset], bins=70, density=True, alpha=0.3, histtype='stepfilled')
axins_left.plot(x, y_gauss, 'k--', linewidth=2)

axins_left.set_xlim(10, 1)
axins_left.set_ylim(1e-3, 1)
axins_left.set_yscale('log')
axins_left.set_xscale('log')
axins_left.set_title("Left Tail", fontsize=9)

# Ticks y formato
ticks = [2, 4, 6, 8, 10]
axins_left.set_xticks(ticks)
axins_left.xaxis.set_major_locator(FixedLocator(ticks))
axins_left.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"-{int(val)}"))
axins_left.xaxis.set_minor_locator(FixedLocator([]))

# Inset: Cola derecha
axins_right = inset_axes(ax, width="100%", height="100%",
                         bbox_to_anchor=(0.66, 0.6, 0.3, 0.3),
                         bbox_transform=ax.transAxes,
                         borderpad=0)

for asset in df_returns.columns:
    axins_right.hist(std_returns[asset], bins=70, density=True, alpha=0.3, histtype='stepfilled')
axins_right.plot(x, y_gauss, 'k--', linewidth=2)

axins_right.set_xlim(1, 10)
axins_right.set_ylim(1e-3, 1)
axins_right.set_yscale('log')
axins_right.set_xscale('log')
axins_right.set_title("Right Tail", fontsize=9)
axins_right.set_xticks(ticks)
axins_right.xaxis.set_major_locator(FixedLocator(ticks))
axins_right.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}"))
axins_right.xaxis.set_minor_locator(FixedLocator([]))

# Pendiente cola derecha
x_manual_right = np.array([1, 10])
y_manual_right = 0.2 * (x_manual_right / 1)**-2.548
axins_right.plot(x_manual_right, y_manual_right, 'r--', label=f'α ≈ -2.548')
axins_right.legend(fontsize=7, loc='upper right')


# Pendiente cola izquierda
x_manual_left = np.array([1, 10])
y_manual_left  = 0.18* (x_manual_left) ** -2.497
axins_left.plot(x_manual_left, y_manual_left, 'r--', label=f'α ≈ -2.497')
axins_left.legend(fontsize=7, loc='upper left')


plt.tight_layout()
plt.show()

# Plot: Cola Izquierda
fig_left, ax_left = plt.subplots(figsize=(6, 4))

for asset in df_returns.columns:
    ax_left.hist(-std_returns[asset], bins=70, density=True, alpha=0.3, histtype='stepfilled')
ax_left.plot(x, y_gauss, 'k--', linewidth=2)

ax_left.set_xlim(10, 1)
ax_left.set_ylim(1e-3, 1)
ax_left.set_yscale('log')
ax_left.set_xscale('log')
ax_left.set_xticks(ticks)
ax_left.xaxis.set_major_locator(FixedLocator(ticks))
ax_left.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"-{int(val)}"))
ax_left.xaxis.set_minor_locator(FixedLocator([]))
ax_left.plot(x_manual_left, y_manual_left, 'r--', label=f'α ≈ -2.497')
ax_left.legend(fontsize=10)
plt.tight_layout()
plt.show()


#Plot: Cola Derecha
fig_right, ax_right = plt.subplots(figsize=(6, 4))

for asset in df_returns.columns:
    ax_right.hist(std_returns[asset], bins=70, density=True, alpha=0.3, histtype='stepfilled')
ax_right.plot(x, y_gauss, 'k--', linewidth=2)

ax_right.set_xlim(1, 10)
ax_right.set_ylim(1e-3, 1)
ax_right.set_yscale('log')
ax_right.set_xscale('log')
ax_right.set_xticks(ticks)
ax_right.xaxis.set_major_locator(FixedLocator(ticks))
ax_right.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}"))
ax_right.xaxis.set_minor_locator(FixedLocator([]))
ax_right.plot(x_manual_right, y_manual_right, 'r--', label=f'α ≈ -2.548')
ax_right.legend(fontsize=10)
plt.tight_layout()
plt.show()
