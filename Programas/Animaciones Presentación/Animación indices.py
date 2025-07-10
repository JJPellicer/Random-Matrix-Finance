import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# -----------------------------------
# Configuración de rutas y activos
# -----------------------------------
assets = [
    'aex', 'bovespa', 'cac40', 'chinaa50', 'copper', 'dax', 'dowjones', 'ftse100',
    'gas', 'gold', 'hangseng', 'ibex35', 'kospi', 'nasdaq', 'nifty50',
    'nikkei225', 'oil', 'omxs30', 'shanghai', 'silver', 'smi', 'sp500', 'spasx200',
    'spbmvipc', 'spmerval', 'sptsx', 'szse', 'us10y'
]
data_path = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos"
output_dir = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Resultados"
output_gif = os.path.join(output_dir, "selected_portfolios_animation_monthly.gif")

# —--------------------------------
# Lectura de datos y log-returns
# ---------------------------------
prices = {}
for asset in assets:
    fp = os.path.join(data_path, f"{asset}.csv")
    df = pd.read_csv(fp, encoding='ISO-8859-1')
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if 'fecha' in c)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    s = (df.iloc[:, 0]
         .astype(str)
         .str.replace('.', '', regex=False)
         .str.replace(',', '.', regex=False))
    prices[asset] = s.astype(float)

df_prices = pd.concat(prices, axis=1).dropna()
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ---------------------------
# Autovalores y portafolios
# ---------------------------
corr = df_returns.corr()
eigenvals, eigenvecs = np.linalg.eigh(corr)
idx = np.argsort(eigenvals)[::-1]
eigenvecs = eigenvecs[:, idx]

port_returns = pd.DataFrame(index=df_returns.index)
for i in range(3):
    w = eigenvecs[:, i]
    w /= np.sum(np.abs(w))
    port_returns[f"Eigen {i+1}"] = df_returns.dot(w)

port_returns['SP500'] = df_returns['sp500']
port_returns['IBEX35'] = df_returns['ibex35']

port_cum = np.exp(port_returns.cumsum())

# --------------------------------
# Generar lista de meses y figura
# --------------------------------
# Lista de meses únicos como frames
months = pd.date_range(port_cum.index.min(), port_cum.index.max(), freq='MS')

columns = ["Eigen 1", "Eigen 2", "Eigen 3", "SP500", "IBEX35"]
fig, ax = plt.subplots(figsize=(10, 6))
lines = {col: ax.plot([], [], label=col)[0] for col in columns}

# Ejes y etiquetas
ax.set_xlim(port_cum.index.min(), port_cum.index.max())
ax.set_ylim(0, port_cum.max().max() * 1.05)
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor Acumulado")

# Ticks Y
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.set_yticks(np.arange(0, port_cum.max().max() * 1.05, 0.5))

# Ticks X anuales (frames son mensuales)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
xticks = [pd.to_datetime(f"{y}-01-01") for y in sorted(set(port_cum.index.year))]
ax.set_xticks(xticks)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.legend(loc="upper left")

# ----------------------------------------
# Init y update para animación mensual
# ----------------------------------------
def init():
    for ln in lines.values():
        ln.set_data([], [])
    return list(lines.values())

def update(frame):
    fecha_actual = months[frame]
    mask = port_cum.index <= fecha_actual
    for name, ln in lines.items():
        ln.set_data(port_cum.index[mask], port_cum[name][mask])
    return list(lines.values())

#----------------------------
# Crear animación y guardar
# ---------------------------
ani = FuncAnimation(fig, update,
                    frames=len(months),
                    init_func=init,
                    interval=50,  # Más rápido que antes
                    blit=False)

writer = PillowWriter(fps=30)
ani.save(output_gif, writer=writer)
print(f"GIF guardado en: {output_gif}")
