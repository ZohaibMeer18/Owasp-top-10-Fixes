import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1️⃣ Load dataset
df = pd.read_excel("dataset.xlsx")
df.columns = df.columns.str.strip()
df['Label'] = df['Label'].str.strip()

# 2️⃣ Select numeric columns
numeric_cols = df.select_dtypes(include='number').columns
X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.clip(-1e9, 1e9)

# 3️⃣ Standardize numeric features
X_scaled = StandardScaler().fit_transform(X)

# 4️⃣ Reduce to 2D using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# 5️⃣ Create DataFrame for plotting
pca_df = pd.DataFrame({
    'PCA1': pca_result[:, 0],
    'PCA2': pca_result[:, 1],
    'Label': df['Label']
})

# 6️⃣ Define colors for each class
classes = ['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
           'DrDoS_NTP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'UDP-lag', 'TFTP']

colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
class_color_map = dict(zip(classes, colors))

# 7️⃣ Plot
plt.figure(figsize=(10,8))

for cls in classes:
    subset = pca_df[pca_df['Label'] == cls]
    if not subset.empty:
        plt.scatter(subset['PCA1'], subset['PCA2'],
                    color=class_color_map[cls],
                    alpha=0.7,
                    label=cls,
                    edgecolors='k',  # optional: black edge for visibility
                    s=50)  # size of points

plt.title('PCA Projection of Multiple Classes')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
