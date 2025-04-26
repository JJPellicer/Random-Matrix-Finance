import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ——————————————————
# 1) Configuración de rutas y lista de activos
# ——————————————————
assets = [
    'aex', 'cac40', 'dax', 'ftse100', 'ibex35', 'omxs30', 'smi',
    'bovespa', 'spmerval',
    'chinaa50', 'shanghai', 'szse',
    'hangseng', 'kospi', 'nikkei225', 'nifty50', 'spasx200',
    'dowjones','nasdaq', 'sp500', 'spbmvipc', 'sptsx',
    'oil', 'gas', 'gold', 'silver', 'copper', 'us10y'
]
data_path  = r"C:\Users\Propietario\Desktop\TFG Juan\Random-Matrix-Finance-main\Datos"
output_dir = r"C:\Users\Propietario\Desktop\TFG Juan\Random-Matrix-Finance-main\Resultados"
output_gif = os.path.join(output_dir, "corr_heatmap_yearly.gif")

# ——————————————————
# 2) Leer datos y calcular rendimientos
# ——————————————————
prices = {}
for asset in assets:
    fp = os.path.join(data_path, f"{asset}.csv")
    df = pd.read_csv(fp, encoding='ISO-8859-1')
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if 'fecha' in c)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    s = df.iloc[:, 0].astype(str)\
          .str.replace('.', '', regex=False)\
          .str.replace(',', '.', regex=False)
    prices[asset] = s.astype(float)

df_prices  = pd.concat(prices, axis=1).dropna()
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ——————————————————
# 3) Lista de años y configuración de la figura
# ——————————————————
years = sorted(df_returns.index.year.unique())

fig, ax = plt.subplots(figsize=(8, 7))
vmin, vmax = -1, 1
# Dibujamos una matriz vacía para inicializar
corr0 = df_returns[df_returns.index.year == years[0]].corr()
im = ax.imshow(corr0, vmin=vmin, vmax=vmax, cmap='coolwarm')
ax.set_xticks(np.arange(len(assets)))
ax.set_yticks(np.arange(len(assets)))
ax.set_xticklabels(assets, rotation=90, fontsize=6)
ax.set_yticklabels(assets, fontsize=6)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ——————————————————
# 4) Función de actualización
# ——————————————————
def update(frame):
    year = years[frame]
    # Correlación del año en curso
    returns_year = df_returns[df_returns.index.year == year]
    corr = returns_year.corr()
    im.set_data(corr.values)
    ax.set_title(f"Matriz de correlación en {year}")
    return (im,)

# ——————————————————
# 5) Crear animación y guardar como GIF
# ——————————————————
ani = FuncAnimation(
    fig, update,
    frames=len(years),
    interval=500,
    blit=False
)
writer = PillowWriter(fps=2)
ani.save(output_gif, writer=writer)
print(f"GIF guardado en: {output_gif}")
