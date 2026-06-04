# Final project - Multivariate Statistics
# Wine dataset
# Ghantarjyan Davit

import json
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix
import statsmodels.api as sm

# ---- load the data ----
wine = load_wine(as_frame=True)
df = wine.frame                       # 13 features + column "target" (cultivar 0,1,2)
features = list(wine.feature_names)

# nicer names for the plots
pretty = {f: f.replace("_", " ").capitalize() for f in features}
pretty["od280/od315_of_diluted_wines"] = "OD280/OD315"

res = {}                              # here I save numbers for the report
res["n_rows"] = int(df.shape[0])
res["n_feats"] = len(features)
res["counts"] = [int((df["target"] == c).sum()) for c in [0, 1, 2]]

# summary table
desc = df[features].describe().T
res["describe"] = [[pretty[f], round(desc.loc[f, "mean"], 2), round(desc.loc[f, "std"], 2),
                    round(desc.loc[f, "min"], 2), round(desc.loc[f, "max"], 2)] for f in features]

colors = ["C0", "C1", "C2"]
labels = ["Cultivar 1", "Cultivar 2", "Cultivar 3"]


# ---- 3. scatter plots ----
pairs = [("flavanoids", "total_phenols"),
         ("flavanoids", "od280/od315_of_diluted_wines"),
         ("alcohol", "proline"),
         ("color_intensity", "hue")]
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
for ax, (a, b) in zip(axs.ravel(), pairs):
    for c in [0, 1, 2]:
        m = df["target"] == c
        ax.scatter(df[a][m], df[b][m], s=15, color=colors[c], label=labels[c])
    ax.set_xlabel(pretty[a])
    ax.set_ylabel(pretty[b])
axs[0, 0].legend(fontsize=8)
fig.suptitle("Scatter plots of some variables (colored by cultivar)")
fig.tight_layout()
fig.savefig("figures/scatter.png", dpi=150)
plt.close(fig)

# ---- 4. normality check ----
gvars = ["flavanoids", "alcohol", "proline", "color_intensity"]
fig, axs = plt.subplots(2, 4, figsize=(12, 5.5))
shap = {}
for j, v in enumerate(gvars):
    x = df[v].values
    W, p = stats.shapiro(x)
    shap[pretty[v]] = p
    axs[0, j].hist(x, bins=20, density=True, edgecolor="black")
    xs = np.linspace(x.min(), x.max(), 100)
    axs[0, j].plot(xs, stats.norm.pdf(xs, x.mean(), x.std()), "r")
    axs[0, j].set_title(pretty[v] + "\nShapiro p=%.1e" % p, fontsize=9)
    stats.probplot(x, dist="norm", plot=axs[1, j])
    axs[1, j].set_title("")
fig.suptitle("Histograms with normal curve (top) and Q-Q plots (bottom)")
fig.tight_layout()
fig.savefig("figures/normality.png", dpi=150)
plt.close(fig)
res["shapiro"] = {k: float(v) for k, v in shap.items()}
res["n_reject"] = int(sum(stats.shapiro(df[f].values)[1] < 0.05 for f in features))

# ---- 5-6. linear regression, response = flavanoids ----
target = "flavanoids"
preds = [f for f in features if f != target]
y = df[target].values
X = StandardScaler().fit_transform(df[preds].values)    # standardize predictors

# correlations of the response with the others (to justify the choice)
corr = df[features].corr()[target].drop(target)
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
res["target"] = pretty[target]
res["top_corr"] = [[pretty[i], round(corr[i], 2)] for i in corr.index[:4]]

model = sm.OLS(y, sm.add_constant(X)).fit()
print(model.summary())
res["r2"] = float(model.rsquared)
res["adj_r2"] = float(model.rsquared_adj)
res["f"] = float(model.fvalue)
res["fp"] = float(model.f_pvalue)
res["df2"] = int(model.df_resid)
names = ["const"] + [pretty[p] for p in preds]
res["coefs"] = [[nm, round(b, 3), round(se, 3), round(t, 2), float(pv)]
                for nm, b, se, t, pv in zip(names, model.params, model.bse,
                                            model.tvalues, model.pvalues)]

# regression diagnostics
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
axs[0].scatter(model.fittedvalues, model.resid, s=15)
axs[0].axhline(0, color="red")
axs[0].set_xlabel("Fitted values")
axs[0].set_ylabel("Residuals")
axs[0].set_title("Residuals vs fitted")
stats.probplot(model.resid, dist="norm", plot=axs[1])
axs[1].set_title("Q-Q plot of residuals")
fig.tight_layout()
fig.savefig("figures/regression.png", dpi=150)
plt.close(fig)

# ---- 7. variable selection (backward elimination by p-value) ----
cols = list(range(len(preds)))
while True:
    m = sm.OLS(y, sm.add_constant(X[:, cols])).fit()
    pvals = m.pvalues[1:]                 # without the constant
    if pvals.max() > 0.05 and len(cols) > 1:
        del cols[int(pvals.argmax())]     # remove the worst variable
    else:
        break
kept = [pretty[preds[c]] for c in cols]
res["kept"] = kept
res["kept_r2"] = float(m.rsquared)
res["dropped"] = len(preds) - len(kept)

# ---- 8. PCA on the 13 standardized variables ----
Z = StandardScaler().fit_transform(df[features].values)
pca = PCA()
scores = pca.fit_transform(Z)
eig = pca.explained_variance_
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
res["eig"] = [round(float(e), 2) for e in eig]
res["ratio"] = [round(float(r) * 100, 1) for r in ratio]
res["cum"] = [round(float(c) * 100, 1) for c in cum]
res["kaiser"] = int((eig > 1).sum())
res["n90"] = int(np.argmax(cum >= 0.90) + 1)

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
axs[0].bar(range(1, len(eig) + 1), eig)
axs[0].axhline(1, color="red")
axs[0].set_xlabel("Component")
axs[0].set_ylabel("Eigenvalue")
axs[0].set_title("Scree plot")
axs[1].plot(range(1, len(eig) + 1), cum * 100, "o-")
axs[1].axhline(90, color="red")
axs[1].set_xlabel("Number of components")
axs[1].set_ylabel("Cumulative %")
axs[1].set_title("Cumulative variance")
fig.tight_layout()
fig.savefig("figures/scree.png", dpi=150)
plt.close(fig)

# ---- 9. correlation circle ----
load = pca.components_.T * np.sqrt(eig)     # correlation of each variable with the PCs
fig, ax = plt.subplots(figsize=(6, 6))
ax.add_artist(plt.Circle((0, 0), 1, color="gray", fill=False))
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
for i, f in enumerate(features):
    ax.arrow(0, 0, load[i, 0], load[i, 1], head_width=0.02, color="blue")
    ax.text(load[i, 0] * 1.1, load[i, 1] * 1.1, pretty[f], fontsize=8, ha="center")
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("PC1 (%.1f%%)" % (ratio[0] * 100))
ax.set_ylabel("PC2 (%.1f%%)" % (ratio[1] * 100))
ax.set_title("Correlation circle")
fig.tight_layout()
fig.savefig("figures/circle.png", dpi=150)
plt.close(fig)
res["pc1_top"] = [pretty[features[i]] for i in np.argsort(-np.abs(load[:, 0]))[:3]]
res["pc2_top"] = [pretty[features[i]] for i in np.argsort(-np.abs(load[:, 1]))[:3]]

# ---- 10. projection of the individuals ----
fig, ax = plt.subplots(figsize=(6.5, 5))
for c in [0, 1, 2]:
    m = df["target"] == c
    ax.scatter(scores[m, 0], scores[m, 1], s=18, color=colors[c], label=labels[c])
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("PC1 (%.1f%%)" % (ratio[0] * 100))
ax.set_ylabel("PC2 (%.1f%%)" % (ratio[1] * 100))
ax.set_title("Individuals on the first two components")
ax.legend()
fig.tight_layout()
fig.savefig("figures/individuals.png", dpi=150)
plt.close(fig)

# ---- 11. extra method: K-means clustering ----
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
res["ari"] = float(adjusted_rand_score(df["target"], km.labels_))
res["sil"] = float(silhouette_score(Z, km.labels_))
cm = confusion_matrix(df["target"], km.labels_)
best = max(sum(cm[i, p[i]] for i in range(3)) for p in permutations(range(3)))
res["acc"] = float(best / cm.sum())

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for c in [0, 1, 2]:
    m = df["target"] == c
    axs[0].scatter(scores[m, 0], scores[m, 1], s=15, color=colors[c], label=labels[c])
axs[0].set_title("Real cultivars")
axs[0].legend(fontsize=8)
axs[0].set_xlabel("PC1")
axs[0].set_ylabel("PC2")
for c in [0, 1, 2]:
    m = km.labels_ == c
    axs[1].scatter(scores[m, 0], scores[m, 1], s=15, label="cluster %d" % (c + 1))
axs[1].set_title("K-means clusters (ARI=%.2f)" % res["ari"])
axs[1].legend(fontsize=8)
axs[1].set_xlabel("PC1")
axs[1].set_ylabel("PC2")
fig.tight_layout()
fig.savefig("figures/kmeans.png", dpi=150)
plt.close(fig)

# save the data and the numbers
df.to_csv("wine_data.csv", index=False)
with open("results.json", "w") as f:
    json.dump(res, f, indent=2)
print("done")
