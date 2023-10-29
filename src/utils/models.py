from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 2
PWD = Path().absolute()

directory = f"{str(PWD)}/src/out"


def train_models(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains a variety of models on training data, and checks their MSE on validation data
    """

    # grab train and validation data
    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    # models are saved as dicts in a list
    models = [
        {"model_type": DummyRegressor, "settings": {}},
        {"model_type": Lasso, "settings": {"alpha": 100, "random_state": RANDOM_STATE}},
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 100, "random_state": RANDOM_STATE},
        },
        {
            "model_type": ElasticNet,
            "settings": {"alpha": 100, "random_state": RANDOM_STATE},
        },
        {"model_type": SVR, "settings": {"degree": 2}},
        {
            "model_type": GradientBoostingRegressor,
            "settings": {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "random_state": RANDOM_STATE,
            },
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 50, "random_state": RANDOM_STATE},
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 15, "random_state": RANDOM_STATE},
        },
        {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": 200, "random_state": RANDOM_STATE},
        },
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 5}},
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 50}},
        {"model_type": KNeighborsRegressor, "settings": {"n_neighbors": 100}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 12}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 50}},
        {"model_type": DecisionTreeRegressor, "settings": {"max_depth": 100}},
    ]

    # intilaize
    model_strings = []
    mse_values_models = []
    clf_vals = []

    # loop over models, train and add values to list
    for mod in models:
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
        clf = mod["model_type"](
            **mod["settings"]  # henter ut settings her med unpacking
        )
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{name}_{settings}")
        clf_vals.append(clf)

    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models],
        }
    )

    # sorter etter mse
    data_models.sort_values(by="mse_values")

    plt.figure(figsize=(10, 10))
    plt.barh(data_models["model_name"], data_models["mse_values"], color="blue")
    plt.xlabel("Mean Error")
    plt.ylabel("Model")
    plt.title("MSE values for different models")
    plt.savefig(f"{PWD}/figs/MANYMODELS_MSE.png")
    plt.show()

    print("MODELS : Done training a variety of models!")


def find_hyper_param(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains RandomForestRegressor with multiple hyperparameters on training data, finds its MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    models = []

    for i in range(1, 252, 50):
        if i == 0:
            i = 1
        model = {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": i, "random_state": RANDOM_STATE},
        }
        models.append(model)

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
        clf = mod["model_type"](**mod["settings"])  # henter ut settings her
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        # finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{name}_{settings}")
        clf_vals.append(clf)

    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [np.sqrt(i) for i in mse_values_models],
        }
    )

    # sort model by mse
    data_models.sort_values(by="mse_values")

    print(data_models)

    plt.figure(figsize=(10, 8))
    barplot = plt.bar(data_models.model_name, data_models.mse_values)
    plt.title("MSE values for RandomForestRegressor")
    plt.xlabel("Model")
    plt.ylabel("Mean Error")
    for idx, rect in enumerate(barplot):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.01 * height,
            round(data_models.mse_values.iloc[idx], 3),
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.savefig(f"{PWD}/figs/MSE_hyperparam_models_V3.png")
    plt.show()

    print("MODELS : Done training hyperparameter models!")


def find_hyper_param_further(split_dict: dict) -> None:
    """
    Input:
        split_dict : split_dict containing x/y train/val/test
    Trains a single model (testing multiple hyperparameters) on test data, finds its MSE on validation data
    """

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]
    X_val = split_dict["x_val"]
    y_val = split_dict["y_val"]

    models = []

    for i in range(151, 252, 30):
        if i == 0:
            i = 1
        model = {
            "model_type": RandomForestRegressor,
            "settings": {"n_estimators": i, "random_state": RANDOM_STATE},
        }
        models.append(model)

    model_strings = []
    mse_values_models = []
    clf_vals = []

    for mod in models:
        name = str(mod["model_type"].__name__)[0:8]
        settings = mod["settings"]

        print(f"MODELS : Training model type: {name}_{settings}")
        clf = mod["model_type"](**mod["settings"])  # henter ut settings her
        clf.fit(X_train, y_train)

        y_predicted = clf.predict(X_val)

        # finn mse
        pf_mse = mean_squared_error(y_val, y_predicted, squared=True)

        mse_values_models.append(pf_mse)
        model_strings.append(f"{name}_{settings}")
        clf_vals.append(clf)

    data_models = pd.DataFrame(
        {
            "model_name": model_strings,
            "mse_values": [sqrt(i) for i in mse_values_models],
        }
    )

    # sort models by mse
    data_models.sort_values(by="mse_values")

    print(data_models)

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(data_models["model_name"], data_models["mse_values"])
    plt.title("MSE values for RandomForestRegressor")
    plt.xlabel("Model")
    plt.ylabel("Mean Error")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(f"{PWD}/figs/MSE_hyperparam_models_further.png")

    print("MODELS : Done training hyperparameter models even further!")

    return


def train_best_model(split_dict: dict, test_data: bool) -> None:
    """
    Trains the model that performed (RandomForestRegressor) best on validation/test data
    """

    if test_data:
        X_chosen = split_dict["x_test"]
        y_chosen = split_dict["y_test"]
    else:
        X_chosen = split_dict["x_val"]
        y_chosen = split_dict["y_val"]

    X_train = split_dict["x_train"]
    y_train = split_dict["y_train"]

    # BEST MODEL:
    best_model = RandomForestRegressor(n_estimators=181, random_state=RANDOM_STATE)

    best_model.fit(X_train, y_train)

    y_test_predicted = best_model.predict(X_chosen)

    test_mse = mean_squared_error(y_chosen, y_test_predicted)
    test_rmse = sqrt(test_mse)

    print(f"MODELS : Model for test data = {test_data}")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)

    importance_df = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": best_model.feature_importances_}
    )

    print(importance_df.sort_values(by="Importance", ascending=False))
