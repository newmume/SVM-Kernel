import numpy as np
from sklearn.svm import SVC


def train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0, degree=3):
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    model.fit(X, y)
    return model


def make_decision_grid(x_range, y_range, resolution=80):
    xx = np.linspace(*x_range, resolution)
    yy = np.linspace(*y_range, resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])
    return XX, YY, grid_points


def compute_decision_surface(model, grid_points):
    return model.decision_function(grid_points)
