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
    'hangseng', 'kospi','nikkei225', 'nifty50','spasx200',
    'dowjones','nasdaq', 'sp500','spbmvipc', 'sptsx',
    'oil', 'gas', 'gold', 'silver', 'copper', 'us10y'
]
data_path  = r"C:\Users\Propietario\Desktop\TFG Juan\Random-Matrix-Finance-main\Datos"
output_dir = r"C:\Users\Propietario\Desktop\TFG Juan\Random-Matrix-Finance-main\Resultados"
output_gif = os.path.join(output_dir, "eigen_portfolios_animation.gif")

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
# 3) Matriz de correlación y autovalores/autovectores
# ——————————————————
corr = df_returns.corr()
eigenvals, eigenvecs = np.linalg.eigh(corr)
idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# ——————————————————
# 4) Construcción de portafolios según los 5 autovectores principales
# ——————————————————
port_returns = pd.DataFrame(index=df_returns.index)
for i in range(5):
    w = eigenvecs[:, i]
    w /= np.sum(np.abs(w))
    port_returns[f"Eigen {i+1}"] = df_returns.dot(w)
# Añadir SP500 como benchmark
if 'sp500' in df_returns.columns:
    port_returns['SP500'] = df_returns['sp500']

# Series de rendimiento acumulado
port_cum = (1 + port_returns).cumprod()

# ——————————————————
# 5) Preparar animación anual
# ——————————————————
years = sorted(df_prices.index.year.unique())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
lines = {col: ax1.plot([], [], label=col)[0] for col in port_cum.columns}

ax1.set_xlim(df_prices.index.min(), df_prices.index.max())
ax1.set_ylim(port_cum.min().min(), port_cum.max().max())
ax1.set_title("Cumulative Performance of Eigen-Portfolios vs SP500")
ax1.legend(loc="upper left")

ax2.set_xlim(port_returns.min().min(), port_returns.max().max())
ax2.set_title("Distribution of Portfolio Returns (log scale)")
ax2.set_yscale('log')

def init():
    for ln in lines.values():
        ln.set_data([], [])
    ax2.clear()
    ax2.set_xlim(port_returns.min().min(), port_returns.max().max())
    ax2.set_title("Distribution of Portfolio Returns (log scale)")
    ax2.set_yscale('log')
    return list(lines.values())

def update(frame):
    year = years[frame]
    mask = port_cum.index.year <= year

    # Actualizar curvas acumuladas
    for name, ln in lines.items():
        ln.set_data(port_cum.index[mask], port_cum[name][mask])

    # Histograma de rendimientos diarios hasta ese año
    ax2.clear()
    data = port_returns.loc[mask].values.flatten()
    ax2.hist(data, bins=30)
    ax2.set_xlim(port_returns.min().min(), port_returns.max().max())
    ax2.set_title(f"Returns up to {year} (log scale)")
    ax2.set_yscale('log')

    return list(lines.values())

ani = FuncAnimation(fig, update,
                    frames=len(years),
                    init_func=init,
                    interval=500,
                    blit=False)

# ——————————————————
# 6) Guardar GIF
# ——————————————————
writer = PillowWriter(fps=2)
ani.save(output_gif, writer=writer)
print(f"GIF guardado en: {output_gif}")
