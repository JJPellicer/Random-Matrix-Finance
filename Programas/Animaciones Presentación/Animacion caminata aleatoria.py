import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# --------------------
# Rutas y archivos
# --------------------
data_path = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos/sp500.csv"
output_dir = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Resultados"
output_gif = os.path.join(output_dir, "sp500_vs_randomwalk.gif")

# ------------------------------------
# Leer SP500 y calcular acumulado
# -----------------------------------
df = pd.read_csv(data_path, encoding='ISO-8859-1')
df.columns = [c.lower().strip() for c in df.columns]
date_col = next(c for c in df.columns if 'fecha' in c)
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
df = df.sort_values(date_col).set_index(date_col)

# Convertir precios a float
s = (df.iloc[:, 0]
     .astype(str)
     .str.replace('.', '', regex=False)
     .str.replace(',', '.', regex=False))
prices = s.astype(float)

# Log-returns y acumulado
log_returns = np.log(prices / prices.shift(1)).dropna()
cum_returns = np.exp(log_returns.cumsum())

# -------------------------------
# Caminata aleatoria discreta
# -------------------------------
n = len(cum_returns)
np.random.seed(1)  # Para reproducibilidad

pasos = np.random.normal(loc=0, scale=0.012, size=n) 
cam_discreta = np.exp(np.cumsum(pasos))
cam_discreta = pd.Series(cam_discreta, index=cum_returns.index)

# --------------------------
# Preparar meses y figura
# --------------------------
months = pd.date_range(cum_returns.index.min(), cum_returns.index.max(), freq='MS')

fig, ax = plt.subplots(figsize=(10, 6))
line_sp, = ax.plot([], [], label='SP500', color='blue')
line_rw, = ax.plot([], [], label='Caminata Aleatoria', color='green', alpha=0.8)

ax.set_xlim(cum_returns.index.min(), cum_returns.index.max())
ax.set_ylim(0, max(cum_returns.max(), cam_discreta.max()) * 1.05)
ax.set_xlabel("Fecha", fontsize=14)
ax.set_ylabel("Valor Acumulado", fontsize=14)
ax.tick_params(axis='both', labelsize=12)  # Ticks más grandes
ax.legend(loc="upper left", fontsize=12)
plt.subplots_adjust(bottom=0.15)


# Ticks
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
xticks = [pd.to_datetime(f"{y}-01-01") for y in sorted(set(cum_returns.index.year))]
ax.set_xticks(xticks)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# ------------------
# Init y Update
# ------------------
def init():
    line_sp.set_data([], [])
    line_rw.set_data([], [])
    return line_sp, line_rw

def update(frame):
    fecha = months[frame]
    mask = cum_returns.index <= fecha
    line_sp.set_data(cum_returns.index[mask], cum_returns[mask])
    line_rw.set_data(cam_discreta.index[mask], cam_discreta[mask])
    return line_sp, line_rw

# -----------
# Animación
# ----------
ani = FuncAnimation(fig, update,
                    frames=len(months),
                    init_func=init,
                    interval=50,
                    blit=True)

writer = PillowWriter(fps=30)
ani.save(output_gif, writer=writer)
print(f"GIF guardado en: {output_gif}")
