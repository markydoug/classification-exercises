import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split


def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


def map_setosa_knn(X_train, y_train, knn, fig_x=12.0, fig_y=5.0):
    plt.rcParams["figure.figsize"] = [fig_x, fig_y]
    plt.rcParams["figure.autolayout"] = True

    n_neighbors = knn.get_params()['n_neighbors']
    weights = knn.get_params()['weights']

    iris = datasets.load_iris()
    X = np.array(X_train[['sepal_length', 'sepal_width']])
    y = y_train.map({'setosa':0, 'versicolor':1, 'virginica':2})
    h = .02

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ['c', 'darkorange', 'darkblue']

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()

    plt.contourf(xx, yy, Z, cmap=cmap_light)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
    palette=cmap_bold, alpha=1.0, edgecolor="black")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("3-Class classification (k = %i, '%s')"
    % (n_neighbors, weights))

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

    plt.show()