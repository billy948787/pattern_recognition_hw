import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

print(len(iris.data))

print(iris.data.shape)

print(iris.feature_names)

print(iris.target_names)

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
    print(f"{iris.feature_names[i]} avg: {features_avg[i]}")

# calculate sd
for data in iris.data:
    for i in range(0, len(iris.feature_names)):
        features_sd[i] += (data[i] - features_avg[i]) ** 2

for i, sd in enumerate(features_sd):
    features_sd[i] /= len(iris.data)
    features_sd[i] = features_sd[i] ** 0.5
    print(f"{iris.feature_names[i]} sd: {features_sd[i]}")