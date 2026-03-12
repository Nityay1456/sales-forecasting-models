# Configuration settings for the sales forecasting models

# Paths
DATA_PATH = 'data/'
MODELS_PATH = 'models/'
OUTPUT_PATH = 'output/'

# Model parameters
ARIMA_PARAMS = {
    'order': (1, 1, 1),
    'seasonal_order': (1, 1, 1, 12)
}

SARIMAX_PARAMS = {
    'order': (1, 1, 1),
    'seasonal_order': (1, 1, 1, 12)
}

# Hyperparameters
HYPERPARAMS = {
    'learning_rate': 0.01,
    'num_iterations': 1000,
    'batch_size': 32
}