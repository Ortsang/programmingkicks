# Lasso and Ridge regression
- mostly these two methods are used to select feactures (selecting related variables)

## linear regression

the linear model minimize the Mean Squared Errors (MSE):
- MSE = sum(Y_{i} - \hat(Y_{i})})^2 / N

for ridge, lasso and ElasticityNet adding a hyperparameter for penelty

- MSE + \lapha \sum a

* ridge - a = a^2
* lasso - a = |a|
* elasticityNet - a = r a^2 + (1-r) |a|


## lasso in python
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
```

- Diabetes data
```python
from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y=True)

features = load_diabetes()['feature_names']
```

- then we can split the dataset into training data and testing data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

- setting up pipeline for the GridSearch 
```python
pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
])
```

- Grid search to get the best alpha
We use neg_mean_squared_error because the grid search tries to maximize the performance metrics, so we add a minus sign to minimize the mean squared error.
```python
search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
```

- get the best alpha and coefficients
- the importance of a feature is the absolute value of its coefficient
    - if the importance is zero then it is discarded
```python
search.best_params_
# {'model__alpha': 1.2000000000000002}

coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
```

    - get the non-zero coefficient
```python
np.array(features)[importance > 0]
# array(['age', 'sex', 'bmi', 'bp', 's1', 's3', 's5'], dtype='<U3')

np.array(features)[importance == 0]
# array(['s2', 's4', 's6'], dtype='<U3')
```

### AIC and BIC for lasso
cited from [this](https://sklearn.apachecn.org/docs/examples/Generalized_Linear_Models/plot_lasso_model_selection.html)

- import libraries
```python
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets
```

- import dataset
```python
# 这样做是为了避免在np.log10时除零
EPSILON = 1e-4

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

rng = np.random.RandomState(42)
X = np.c_[X, rng.randn(X.shape[0], 14)]  # 添加一些不好的特征

# 将最小角度回归得到的数据标准化，以便进行比较
X /= np.sqrt(np.sum(X ** 2, axis=0))
```

- using AIC and BIC
```python
# LassoLarsIC: 用BIC/AIC判据进行最小角度回归

model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_
```

- visualization 
```python
def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_ + EPSILON
    alphas_ = model.alphas_ + EPSILON
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s 判据' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s 估计' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('判据')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('模型选择的信息判据 (训练时间:%.3fs)'
          % t_bic)
```




## 引用
- [Lasso模型选择:交叉验证 / AIC / BIC](https://sklearn.apachecn.org/docs/examples/Generalized_Linear_Models/plot_lasso_model_selection.html)
