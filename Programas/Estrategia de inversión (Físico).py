import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------
# Configuración y rutas de activos
# ---------------------------------------------
assets = [
    'szse', 'shanghai', 'chinaa50', 'nifty50', 'hangseng', 'kospi', 'nikkei225', 'spasx200',
    'ftse100', 'aex', 'cac40', 'dax', 'ibex35', 'omxs30', 'smi',
    'sptsx', 'sp500', 'dowjones', 'us10y', 'nasdaq', 'spbmvipc',
    'bovespa', 'spmerval',
    'copper', 'silver', 'gold',
    'oil', 'gas'
]

data_path  = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Datos"
output_dir = r"C:/Users/Juan/Documents/GitHub/Random-Matrix-Finance/Resultados"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------
# Lectura de datos y log-returns
# ---------------------------------------------
prices = {}
for asset in assets:
    fp = os.path.join(data_path, f"{asset}.csv")
    if not os.path.exists(fp):
        print(f"⚠️ No encontrado: {asset}")
        continue
    df = pd.read_csv(fp, encoding='ISO-8859-1')
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next(c for c in df.columns if 'fecha' in c)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(date_col).set_index(date_col)
    s = (
        df.iloc[:, 0].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )
    prices[asset] = s

df_prices  = pd.concat(prices, axis=1).dropna()
df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ---------------------------------------------
# Rebalanceo anual con market mode (todo positivo)
# ---------------------------------------------
years = sorted(df_returns.index.year.unique())
portfolio_returns = pd.Series(dtype=float)

for i, year in enumerate(years):
    if i == 0:
        w = np.repeat(1/len(df_returns.columns), len(df_returns.columns))
    else:
        prev_year = years[i - 1]
        ret_prev = df_returns[df_returns.index.year == prev_year]
        if len(ret_prev) < 50:
            w = np.repeat(1/len(df_returns.columns), len(df_returns.columns))
        else:
            corr = ret_prev.corr()
            eigvals, eigvecs = np.linalg.eigh(corr.values)
            market_mode = np.abs(eigvecs[:, eigvals.argmax()])
            w = market_mode / market_mode.sum()

    ret_year = df_returns[df_returns.index.year == year]
    port_ret = ret_year.dot(w)
    portfolio_returns = pd.concat([portfolio_returns, port_ret])

# ---------------------------------------------
#1 Portfolio equiponderado
# ---------------------------------------------
activos_europeos = ['ftse100', 'aex', 'cac40', 'dax', 'ibex35', 'omxs30', 'smi', 'sp500']
equal_weights = np.repeat(1 / len(activos_europeos), len(activos_europeos))
equal_weight_returns = df_returns[activos_europeos].dot(equal_weights)

# ---------------------------------------------
# Plot: comparación con activos europeos
# ---------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

for asset in activos_europeos:
    if asset not in df_returns.columns:
        print(f"⚠️ Activo no disponible: {asset}")
        continue
    asset_cum = df_returns[asset].cumsum().apply(np.exp)
    asset_cum = asset_cum.loc[portfolio_returns.index]
    asset_cum /= asset_cum.iloc[0]
    ax.plot(asset_cum.index, asset_cum, alpha=0.2, linewidth=1, label=asset)

cum_port = portfolio_returns.cumsum().apply(np.exp)
cum_port /= cum_port.iloc[0]
ax.plot(cum_port.index, cum_port, color='red', linewidth=2.5, label='Auto‑portfolio anual (market mode)')

cum_eq = equal_weight_returns.cumsum().apply(np.exp)
cum_eq /= cum_eq.iloc[0]
ax.plot(cum_eq.index, cum_eq, color='blue', linewidth=2.5, label='Portfolio equiponderado')

ax.set_ylabel('Multiplicador de rendimiento acumulado',fontsize=15)
ax.set_xlabel('Fecha',fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(loc='upper left', fontsize='small', ncol=2, frameon=False)
ax.grid(False)
plt.tight_layout()

fig_path = os.path.join(output_dir, "comparacion_activos_europeos.png")
fig.savefig(fig_path, dpi=150)
plt.show()
print(f"📈 Gráfico guardado en: {fig_path}")

# ---------------------------------------------
# Crear DataFrame acumulado para animación
# ---------------------------------------------
df_plot = pd.DataFrame(index=portfolio_returns.index)
df_plot['Market Mode'] = cum_port
df_plot['Equally Weighted'] = cum_eq

for asset in activos_europeos:
    asset_cum = df_returns[asset].cumsum().apply(np.exp)
    asset_cum = asset_cum.loc[df_plot.index]
    asset_cum /= asset_cum.iloc[0]
    df_plot[asset] = asset_cum

df_plot = df_plot.loc[df_plot.first_valid_index():]

# ---------------------------------------------
# Animación mensual de los activos y cartera Market Mode
# ---------------------------------------------
months = pd.date_range(df_plot.index.min(), df_plot.index.max(), freq='MS')

fig, ax = plt.subplots(figsize=(12, 7))
lines = {}

for col in df_plot.columns:
    if col == "Market Mode":
        lines[col], = ax.plot([], [], color='red', linewidth=2.5, label=col)
    elif col == "Equally Weighted":
        lines[col], = ax.plot([], [], color='blue', linewidth=2.5, linestyle='--', label=col)
    else:
        lines[col], = ax.plot([], [], alpha=0.2, linewidth=1, label=col)

ax.set_xlim(df_plot.index.min(), df_plot.index.max())
ax.set_ylim(0.95, df_plot.max().max() * 1.05)
ax.set_xlabel("Fecha")
ax.set_ylabel("Multiplicador de rendimiento acumulado")
ax.set_title("Evolución mensual del auto‑portfolio vs índices europeos")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
xticks = [pd.to_datetime(f"{y}-01-01") for y in sorted(set(df_plot.index.year))]
ax.set_xticks(xticks)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.legend(loc="upper left", fontsize="small", ncol=2, frameon=False)

def init():
    for ln in lines.values():
        ln.set_data([], [])
    return list(lines.values())

def update(frame):
    fecha = months[frame]
    mask = df_plot.index <= fecha
    for col, ln in lines.items():
        ln.set_data(df_plot.index[mask], df_plot[col][mask])
    return list(lines.values())

output_gif = os.path.join(output_dir, "animacion_market_mode_vs_europeos.gif")
ani = FuncAnimation(fig, update, frames=len(months), init_func=init, interval=50, blit=False)
writer = PillowWriter(fps=30)
ani.save(output_gif, writer=writer)
print(f"🎞️ GIF guardado en: {output_gif}")


# ---------------------------------------------------------
# Animación del histograma de autovalores anuales (Market Mode Rojo)
# ---------------------------------------------------------
from matplotlib.animation import FuncAnimation

# Obtener años válidos con suficientes datos
years_valid = [y for y in years if len(df_returns[df_returns.index.year == y]) >= 50]

eigval_dict = {}
for year in years_valid:
    ret_year = df_returns[df_returns.index.year == year]
    corr = ret_year.corr()
    eigvals = np.linalg.eigvalsh(corr)
    eigval_dict[year] = eigvals

# Crear figura
fig, ax = plt.subplots(figsize=(16, 10))
bins = np.linspace(0, max(max(v) for v in eigval_dict.values()) * 1.1, 50)
bar_container = ax.hist([], bins=bins, color='gray', edgecolor='black', linewidth=0.8)[2]


ax.set_xlim(bins[0], bins[-1])
ax.set_ylim(0, 20)
ax.set_xlabel("Autovalor",fontsize=15)
ax.set_ylabel("Frecuencia",fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
title = ax.set_title("")

# Init de animación
def init_hist():
    for bar in bar_container:
        bar.set_height(0)
        bar.set_color('gray')
        bar.set_edgecolor('black')
    return bar_container

# Update de animación
def update_hist(frame):
    year = years_valid[frame]
    eigvals = eigval_dict[year]
    counts, _ = np.histogram(eigvals, bins=bins)

    max_val = eigvals.max()
    # Encontrar la barra que contiene el market mode
    bin_index = np.digitize([max_val], bins)[0] - 1

    for i, bar in enumerate(bar_container):
        bar.set_height(counts[i])
        bar.set_color('red' if i == bin_index else 'gray')

    title.set_text(f"Distribución de autovalores - Año {year}")
    title.set_fontsize(18)
    return bar_container

# Guardar animación
output_gif_hist = os.path.join(output_dir, "histograma_autovalores_anual.gif")
ani_hist = FuncAnimation(fig, update_hist, frames=len(years_valid),
                         init_func=init_hist, interval=100, blit=False)

writer = PillowWriter(fps=2)
ani_hist.save(output_gif_hist, writer=writer)
print(f"📊 Histograma animado guardado en: {output_gif_hist}")
