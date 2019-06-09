import os

from collections import OrderedDict

import pandas as pd

import numpy as np
import scipy as sp

from sklearn import datasets
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt



# Outputs

output_path = os.path.expanduser("~/plots")  # M:/plots on windows
if not os.path.exists(output_path):
  os.makedirs(output_path)



def show_axes_legend(a, loc="upper right"):
  """Create legend based on labeled objects in the order they were labeled.

  Args:
    a (matplotlib.axes.Axes): target axes
    loc (str): axes location (default: "upper right")
  """

  handles, labels = a.get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  legend = a.legend(
    by_label.values(), by_label.keys(),
    borderpad=0.5,
    borderaxespad=0,
    fancybox=False,
    edgecolor="black",
    framealpha=1,
    loc=loc,
    fontsize="x-small",
    ncol=1
  )
  frame = legend.get_frame().set_linewidth(0.75)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Args:
      x (np.ndarray): data to base x-axis meshgrid on
      y (np.ndarray): data to base y-axis meshgrid on
      h (float): stepsize for meshgrid, optional

    Returns:
      xx (np.ndarray): x axis of meshgrid
      yy (np.ndarray): y axis of meshgrid
    """

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(
      np.arange(x_min, x_max, h),
      np.arange(y_min, y_max, h)
    )
    return xx, yy


# Load iris dataset (see Fisher's Iris dataset)

iris = datasets.load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])

color_map = plt.get_cmap("Set1")
colors = list(color_map.colors)
labels = [label.capitalize() for label in iris.target_names]



# Scatterplot matrix

f, a = plt.subplots(1, 1, dpi=300, figsize=(10, 10))

axes = pd.scatter_matrix(df, ax=a)

for i in np.arange(0, X.shape[1]):
  for j in np.arange(0, X.shape[1]):
    axes[i, j].xaxis.set_tick_params(rotation=0)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"iris_scatterplot_matrix.png"), bbox_inches="tight")
plt.close(f)



# Scatterplot matrix (stratified)

f, a = plt.subplots(1, 1, dpi=300, figsize=(10, 10))
axes = pd.plotting.scatter_matrix(df, ax=a, diagonal=None, color=[colors[k] for k in y])

for i in np.arange(0, X.shape[1]):

  for k in np.unique(y):
    x = X[y == k, i]
    kde = sp.stats.gaussian_kde(x)
    ind = np.linspace(x.min(), x.max(), 1000)
    axes[i, i].plot(ind, kde.evaluate(ind), linewidth=1, color=colors[k], label=labels[k])

  show_axes_legend(axes[i, i])

  for j in np.arange(0, X.shape[1]):
    axes[i, j].xaxis.set_tick_params(rotation=0)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"iris_scatterplot_matrix_stratified.png"), bbox_inches="tight")
plt.close(f)



# Predict setosa based on petal length and sepal width (linear SVM)

X_train = X[:, (2, 1)]
y_train = (y == 0).astype(int)

model = LinearSVC(C=1.0)
model.fit(X_train, y_train)

f, a = plt.subplots(1, 1, dpi=300, figsize=(5, 5))

xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], 0.01)
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
a.contourf(xx, yy, z, cmap=color_map, alpha=0.1)
a.contour(xx, yy, z, linewidths=0.5, colors="black")

for k in np.unique(y_train):
  a.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1], s=3, color=colors[k], label="Setosa" if k == 1 else "Other")

a.set_xlabel("Petal Length")
a.set_ylabel("Sepal Width")

show_axes_legend(a)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"model_svm_scatterplot_1.png"), bbox_inches="tight")
plt.close(f)


# Predict setosa based on petal length and sepal width (linear SVM)

X_train = X[:, (2, 1)]
y_train = y

model = LinearSVC(C=1.0)
model.fit(X_train, y_train)

f, a = plt.subplots(1, 1, dpi=300, figsize=(5, 5))

xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], 0.01)
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
a.contourf(xx, yy, z, cmap=color_map, alpha=0.1)
a.contour(xx, yy, z, linewidths=0.5, colors="black")

for k in np.unique(y_train):
  a.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1], s=3, color=colors[k], label=labels[k])

a.set_xlabel("Petal Length")
a.set_ylabel("Sepal Width")

show_axes_legend(a)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"model_svm_scatterplot_2.png"), bbox_inches="tight")
plt.close(f)



# Predict setosa based on sepal length and sepal width (linear SVM)

X_train = X[:, (0, 1)]
y_train = y

model = LinearSVC(C=1.0)
model.fit(X_train, y_train)

f, a = plt.subplots(1, 1, dpi=300, figsize=(5, 5))

xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], 0.01)
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
a.contourf(xx, yy, z, cmap=color_map, alpha=0.1)
a.contour(xx, yy, z, linewidths=0.5, colors="black")

for k in np.unique(y_train):
  a.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1], s=3, color=colors[k], label=labels[k])

a.set_xlabel("Petal Length")
a.set_ylabel("Sepal Width")

show_axes_legend(a)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"model_svm_scatterplot_3.png"), bbox_inches="tight")
plt.close(f)



# Predict setosa based on sepal length and sepal width (SVM with RBF)

X_train = X[:, (0, 1)]
y_train = y

model = SVC(kernel="rbf", C=1.0, gamma=0.7)
model.fit(X_train, y_train)

f, a = plt.subplots(1, 1, dpi=300, figsize=(5, 5))

xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], 0.01)
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
a.contourf(xx, yy, z, cmap=color_map, alpha=0.1)
a.contour(xx, yy, z, linewidths=0.5, colors="black")

for k in np.unique(y_train):
  a.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1], s=3, color=colors[k], label=labels[k])

a.set_xlabel("Petal Length")
a.set_ylabel("Sepal Width")

show_axes_legend(a)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"model_svm_scatterplot_4.png"), bbox_inches="tight")
plt.close(f)



# Predict setosa based on sepal length and sepal width (logistic regression)

X_train = X[:, (0, 1)]
y_train = y

model = LogisticRegression()
model.fit(X_train, y_train)

f, a = plt.subplots(1, 1, dpi=300, figsize=(5, 5))

xx, yy = make_meshgrid(X_train[:, 0], X_train[:, 1], 0.01)
z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
a.contourf(xx, yy, z, cmap=color_map, alpha=0.1)
a.contour(xx, yy, z, linewidths=0.5, colors="black")

for k in np.unique(y_train):
  a.scatter(X_train[y_train == k, 0], X_train[y_train == k, 1], s=3, color=colors[k], label=labels[k])

a.set_xlabel("Petal Length")
a.set_ylabel("Sepal Width")

show_axes_legend(a)

plt.tight_layout()
f.savefig(os.path.join(output_path, f"model_svm_scatterplot_5.png"), bbox_inches="tight")
plt.close(f)


