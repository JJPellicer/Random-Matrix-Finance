import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ——————————————————
# 1) Configuración de rutas y lista de activos
# ——————————————————
assets = [
    'aex', 'bovespa', 'cac40', 'chinaa50', 'copper', 'dax', 'dowjones', 'ftse100',
    'gas', 'gold', 'hangseng', 'ibex35', 'kospi', 'nasdaq', 'nifty50',
    'nikkei225', 'oil', 'omxs30', 'shanghai', 'silver', 'smi', 'sp500', 'spasx200',
    'spbmvipc', 'spmerval', 'sptsx', 'szse', 'us10y'
]
data_path  = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos"
output_dir = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Resultados"
output_gif = os.path.join(output_dir, "assets_animation_annual.gif")

# ——————————————————
# 2) Lectura y combinación de precios
# ——————————————————
prices = {}
for asset in assets:
    fp = os.path.join(data_path, f"{asset}.csv")
    df = pd.read_csv(fp, encoding='ISO-8859-1')
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if 'fecha' in c)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    s = df.iloc[:, 0].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    prices[asset] = s.astype(float)

df_prices  = pd.concat(prices, axis=1).dropna()
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ——————————————————
# 3) Preparar lista de años y figura
# ——————————————————
years = sorted(df_prices.index.year.unique())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
lines = {col: ax1.plot([], [], label=col)[0] for col in df_prices.columns}

ax1.set_xlim(df_prices.index.min(), df_prices.index.max())
ax1.set_ylim(df_prices.min().min(), df_prices.max().max())
ax1.set_title("Movimientos de Precios de los Assets (acumulado por año)")
ax1.legend(loc="upper left")

ax2.set_xlim(df_returns.min().min(), df_returns.max().max())
ax2.set_title("Distribución de Rendimientos (acumulado por año)")

# ——————————————————
# 4) Funciones init y update
# ——————————————————
def init():
    for ln in lines.values():
        ln.set_data([], [])
    ax2.clear()
    ax2.set_xlim(df_returns.min().min(), df_returns.max().max())
    ax2.set_title("Distribución de Rendimientos (acumulado por año)")
    return list(lines.values())

def update(frame):
    year = years[frame]
    mask = df_prices.index.year <= year

    # Actualizar precios
    for name, ln in lines.items():
        ln.set_data(df_prices.index[mask], df_prices[name][mask])

    # Histograma de rendimientos hasta ese año
    ax2.clear()
    hist_data = df_returns[df_returns.index.year <= year].values.flatten()
    ax2.hist(hist_data, bins=30)
    ax2.set_xlim(df_returns.min().min(), df_returns.max().max())
    ax2.set_title(f"Rendimientos log hasta {year}")
    return list(lines.values())

# ——————————————————
# 5) Crear animación anual y guardar
# ——————————————————
ani = FuncAnimation(
    fig, update,
    frames=len(years),
    init_func=init,
    interval=500,    # medio segundo por año
    blit=False
)

writer = PillowWriter(fps=2)  # 2 años por segundo
ani.save(output_gif, writer=writer)
print(f"GIF anual guardado en: {output_gif}")
