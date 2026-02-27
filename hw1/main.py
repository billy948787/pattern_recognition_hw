import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
from sklearn import datasets
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

print(f"Number of samples: {len(iris.data)}, Number of features: {len(iris.feature_names)}")

print("Classes: ", iris.target_names)

class_sample_counts = [0] * len(iris.target_names)

for x in iris.target:
    class_sample_counts[x] += 1

for i in range(0, 3):
    print(f"{iris.target_names[i]} has {class_sample_counts[i]} samples")

features_avg = [0] * len(iris.feature_names)
features_sd = [0] * len(iris.feature_names)

# calculate avg
for data in iris.data:
    for i in range(0, len(iris.feature_names)):
        features_avg[i] += data[i]

for i, avg in enumerate(features_avg):
    features_avg[i] /= len(iris.data)

# calculate sd
for data in iris.data:
    for i in range(0, len(iris.feature_names)):
        features_sd[i] += (data[i] - features_avg[i]) ** 2

for i, sd in enumerate(features_sd):
    features_sd[i] /= (len(iris.data) - 1)
    features_sd[i] = features_sd[i] ** 0.5
print("\nFeature Statistics")
print("-" * 42)
print(f"{'Feature':<18} {'Avg':>8} {'SD':>8}")
print("-" * 42)

for i in range(len(iris.feature_names)):
    print(f"{iris.feature_names[i]:<18} {features_avg[i]:>8.3f} {features_sd[i]:>8.3f}")

print("-" * 42)

# features_avg = np.mean(iris.data, axis=0) 
# features_sd = np.std(iris.data, axis=0, ddof=1)

# for i in range(len(iris.feature_names)):
#     print(f"{iris.feature_names[i]} avg: {features_avg[i]}")
#     print(f"{iris.feature_names[i]} sd: {features_sd[i]}")

# scatter plot matrix
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = pd.Series(iris.target)

sns.set_theme(style="ticks")
sns.pairplot(
    df,
    hue="target",
    diag_kind="hist",
    palette=["#0095ff", "#1ac51a", "#ff0000"],
    corner=False
)
plt.tight_layout()
plt.show(block = False)

# two of most relevant features 
(x, y) = (2, 3)

plt.figure(figsize=(5, 5))
scatter = plt.scatter(iris.data[:, x], iris.data[:, y], c=iris.target, cmap=ListedColormap(["#0095ff", "#1ac51a", "#ff0000"]))
plt.xlabel(iris.feature_names[x])
plt.ylabel(iris.feature_names[y])
plt.title(f"Scatter plot of {iris.feature_names[x]} vs {iris.feature_names[y]}")
handles, _ = scatter.legend_elements()
plt.legend(handles, iris.target_names)
plt.show(block = False)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42, stratify=iris.target)

print(f"Train set size: {len(X_train)}")

train_target_counts = [0] * len(iris.target_names)

for x in y_train:
    train_target_counts[x] += 1 

for i, count in enumerate(train_target_counts):
    print(f"    {iris.target_names[i]} : {count} ("f"{(count / len(X_train)) * 100:.2f}%)")

print(f"Test set size: {len(X_test)}")
test_target_counts = [0] * len(iris.target_names)
for x in y_test:
    test_target_counts[x] += 1
for i, count in enumerate(test_target_counts):
    print(f"    {iris.target_names[i]} : {count} ("f"{(count / len(X_test)) * 100:.2f}%)")