This folder contains examples and user specified data files.

The diabetes data set is provided in different formats.
It was taken from https://www.openml.org/search?type=data&sort=runs&id=41519&status=active

Note: The column "class" was renamed to "target".

Number of Instances: 442
Number of Attributes: First 10 columns are numeric predictive values
Target: Column 11 is a quantitative measure of disease progression one year after baseline

It is available in the following formats:

- csv: diabetes.csv
- pkl: diabetes.pkl

Furthermore, it is available in spotPython as a torch DataSet, see:
https://sequential-parameter-optimization.github.io/spotPython/reference/spotPython/data/diabetes/


Moons data set:
The moons data set is a synthetic data set, which is used to test clustering algorithms.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
n_features = 2
n_samples = 500
target_column = "y"
ds =  make_moons(n_samples, noise=0.5, random_state=0)
X, y = ds
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
test = pd.DataFrame(np.hstack((X_test, y_test.reshape(-1, 1))))
train.columns = [f"x{i}" for i in range(1, n_features+1)] + [target_column]
test.columns = [f"x{i}" for i in range(1, n_features+1)] + [target_column]
train.head()
# combine the training and test data and save to a csv file
data = pd.concat([train, test])
data.to_csv('moon.csv', index=False)