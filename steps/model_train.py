import logging
import mlflow

import pandas as pd
from zenml import step
from src.modeldev import LinearRegrssionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train:pd.DataFrame,x_test:pd.DataFrame
                ,y_train:pd.DataFrame,y_test:pd.DataFrame,
                config:ModelNameConfig,) ->RegressorMixin:
    try:
        model=None
        if config.model_name=="LinearRegression":
            mlflow.sklearn.autolog()
            model=LinearRegrssionModel()
            trained_model=model.train(x_train,y_train)
            return trained_model
        
        else:
            raise ValueError(config.model_name)
        
    except Exception as e:
        logging.error(e)
        raise e