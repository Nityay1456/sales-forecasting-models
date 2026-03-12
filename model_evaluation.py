import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error 


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: Dictionary containing MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def compare_models(models, X, y):
    """
    Compare the performance of different models.

    Parameters:
    models (dict): Dictionary of model names and model instances.
    X (array-like): Features.
    y (array-like): Target variable.

    Returns:
    dict: Dictionary containing the evaluation metrics for each model.
    """
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)
        results[name] = evaluate_model(y, predictions)

    return results
