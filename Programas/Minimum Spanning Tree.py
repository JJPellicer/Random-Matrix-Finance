# Minimum Spanning Tree (MST) con nodos coloreados por tipo de activo y tamaño según centralidad

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
import os

# Lista de activos
assets = [
    'aex', 'bovespa', 'cac40', 'chinaa50', 'dax', 'dowjones', 'ftse100',
    'hangseng', 'ibex35', 'kospi', 'nasdaq', 'nifty50',
    'nikkei225', 'omxs30', 'shanghai', 'smi', 'sp500', 'spasx200',
    'spbmvipc', 'spmerval', 'sptsx', 'szse', 'oil', 'gas', 'gold', 'silver', 'copper', 'us10y'
]

# Clasificación de activos (para colorear)
activos_tipo = {
    'bonos': ['us10y'],
    'commodities': ['gold', 'silver', 'copper', 'oil', 'gas'],
    'mercados': list(set(assets) - set(['us10y', 'gold', 'silver', 'copper', 'oil', 'gas']))
}

# Asignar color por tipo
colores = {'bonos': 'gold', 'commodities': 'brown', 'mercados': 'skyblue'}
color_nodos = []
for asset in assets:
    for tipo, lista in activos_tipo.items():
        if asset in lista:
            color_nodos.append(colores[tipo])

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

# Calcular matriz de correlación
correlation_matrix = df_returns.corr()

# Convertir correlación en distancia: d_ij = sqrt(2(1 - rho_ij))
dist_matrix = np.sqrt(2 * (1 - correlation_matrix))

# Aplicar MST
mst = minimum_spanning_tree(dist_matrix.values).toarray()

# Crear grafo con NetworkX
G = nx.Graph()
for i, name in enumerate(correlation_matrix.columns):
    G.add_node(name)

n = len(correlation_matrix.columns)
for i in range(n):
    for j in range(n):
        if mst[i, j] > 0:
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=mst[i, j])

# Calcular centralidad para escalar nodos
centralidad = nx.degree_centrality(G)
tamanos_nodos = [300 + 1500 * centralidad[n] for n in G.nodes()]

# Dibujar grafo con layout mejorado
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.kamada_kawai_layout(G)

nx.draw(G, pos,
        with_labels=True,
        node_color=color_nodos,
        edge_color='gray',
        node_size=tamanos_nodos,
        font_size=9,
        width=1.5,
        ax=ax)

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos,
    edge_labels={k: f"{v:.2f}" for k, v in labels.items()},
    font_size=7,
    ax=ax)

plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
plt.show()

# -------------------------
# HEATMAP ORDENADO 
# -------------------------

order_visual = [
    'szse', 'shanghai', 'chinaa50', 'nifty50', 'hangseng', 'kospi', 'nikkei225', 'spasx200',
    'ftse100', 'aex', 'cac40', 'dax', 'ibex35', 'omxs30', 'smi',
    'sptsx', 'sp500', 'dowjones', 'us10y', 'nasdaq', 'spbmvipc',
    'bovespa', 'spmerval',
    'copper', 'silver', 'gold',
    'oil', 'gas'
]

# Generamos el heatmap ordenado según el orden visual
C_ord_visual = correlation_matrix.reindex(index=order_visual, columns=order_visual)

plt.figure(figsize=(8, 7))
sns.heatmap(C_ord_visual, cmap='RdBu_r', center=0,
            xticklabels=order_visual,
            yticklabels=order_visual)
plt.tight_layout()
plt.show()



