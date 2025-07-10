# Random Matrix Finance

This repository contains the code, data, and results of the Bachelor's Thesis titled **"Application of Random Matrix Theory to the Study of Correlations in Financial Markets"**. The project applies concepts from statistical physics to finance, falling within the field of **Sociophysics** or **Econophysics**.

## Objective

The aim of this project is to analyze correlations between different financial assets using **Random Matrix Theory (RMT)**, with a focus on:

- The structure of the eigenvalue spectrum.
- Identifying non-random patterns in the market.
- Constructing optimized investment portfolios based on these structures.

---

## Repository Structure

```
Random-Matrix-Finance/
│
├── Datos/                          # Raw financial data (indices, assets, etc.)
│
├── Programas/                      # Main analysis scripts and notebooks
│   ├── Autovalores y autovectores
│   ├── Gaussiana vs datos
│   ├── Minimum Spanning Tree
│   ├── Resultados sp500
│   └── Animaciones Presentación/
│       ├── Animacion caminata aleatoria
│       ├── Animacion Heatmap
│       └── Animación indices
│
├── Resultados/                     # Output results (plots, histograms, heatmaps, etc.)
│
├── Tratamiento datos/             # Data preprocessing and formatting scripts
│
├── LICENSE                         # Project license
├── README.md                       # This file
└── .gitignore                      # Git ignored files
```

---

## Requirements

This project was developed using:

- Python 3.10+
- Main libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `networkx`
  - `yfinance`

To install all dependencies:

```bash
pip install -r requirements.txt
```

*(If the file is missing, you can create it with `pip freeze > requirements.txt`)*

---

## How to Run

You can explore the main notebooks under `Programas/` depending on the analysis you are interested in:

- **Autovalores y autovectores**: Eigenvalue analysis of the correlation matrix.
- **Gaussiana vs datos**: Comparison between empirical and Gaussian distributions.
- **Minimum Spanning Tree**: Structural representation of asset relationships.
- **Resultados sp500**: Specific study of the S&P 500 index.
- **Animaciones Presentación**: Visual scripts used for presentation animations.

---

## Author

This project was developed by **Juan José Pellicer Querol** as part of the Physics Degree at the **University of Barcelona (UB)**.

## Bibliography

- Mantegna, R. N., & Stanley, H. E. (1999). *An Introduction to Econophysics: Correlations and Complexity in Finance*. Cambridge University Press.
- Bouchaud, J. P., & Potters, M. (2015). *Financial applications of random matrix theory: a short review*.
- Cheng, H. J. Z., Rea, W., & Rea, A. (2015). A comparison of three network portfolio methods: Evidence from the Dow Jones. *arXiv preprint* arXiv:1512.01905.
- Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance, 1*(2), 223–236.
- Cheng, H. J. Z., Rea, W., & Rea, A. (2015). An application of correlation clustering to portfolio diversification. *arXiv preprint* arXiv:1511.07945.
- Mantegna, R. N. (1999). Hierarchical structure in financial markets. *The European Physical Journal B-Condensed Matter and Complex Systems*, 11, 193–197.
- Mantegna, R. N. (2023). *Noise and information in economic and financial systems* [Video]. YouTube. [https://www.youtube.com/watch?v=zUI7_VTloxA](https://www.youtube.com/watch?v=zUI7_VTloxA)
- Bouchaud, J.-P., & Potters, M. (2003). *Theory of Financial Risk and Derivative Pricing: From Statistical Physics to Risk Management*. Cambridge University Press.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
