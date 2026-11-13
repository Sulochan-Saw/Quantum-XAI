
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def get_models(seed=0):
    """
    Return all classical baseline models.
    """

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=1,
            random_state=seed
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.03,
            random_state=seed,
            objective="reg:squarederror",
            n_jobs=-1
        ),

        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=0.01,
            max_iter=2000,
            random_state=seed
        )
    }

    return models
