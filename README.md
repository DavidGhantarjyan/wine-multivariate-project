# Wine — Multivariate Statistics Project

## Dataset
The **Wine** dataset (UCI Machine Learning Repository, also available in
scikit-learn as `load_wine`): 178 wines, 13 numeric chemical variables and
3 grape cultivars. A copy is saved in [`wine_data.csv`](wine_data.csv).

## How to reproduce
```bash
pip install -r requirements.txt
python analysis.py
```
Running the script recreates every figure in `figures/`, prints the regression
summary and saves the numeric results to `results.json`.

## What `analysis.py` does
1. Loads and describes the data.
2. Scatter plots of selected variable pairs (colored by cultivar).
3. Normality check — histograms, Q-Q plots and the Shapiro–Wilk test.
4. Linear regression with response = **Flavanoids**.
5. Variable selection by backward elimination (p-value).
6. PCA on the 13 standardized variables (scree plot).
7. Correlation circle.
8. Projection of the individuals on the first two principal components.
9. Extra method: K-means clustering compared with the true cultivars.

## Figures
| File | Content |
|------|---------|
| `figures/scatter.png` | scatter plots colored by cultivar |
| `figures/normality.png` | histograms + Q-Q plots |
| `figures/regression.png` | regression diagnostics |
| `figures/scree.png` | PCA scree + cumulative variance |
| `figures/circle.png` | correlation circle |
| `figures/individuals.png` | individuals on PC1–PC2 |
| `figures/kmeans.png` | K-means clusters vs real cultivars |
