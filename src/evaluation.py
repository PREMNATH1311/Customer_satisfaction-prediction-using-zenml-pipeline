import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_score(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse=mean_squared_error(y_true,y_pred)
            logging.info(f"MSE {mse}")
            return mse
        except Exception as e:
            logging.error(e)
            raise e
        
class R2(Evaluation):
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating R2 scores")
            r2=r2_score(y_true,y_pred)
            logging.info(f"R2_score {r2}")
            return r2
        except Exception as e:
            logging.error()
            raise e
        
class RMSE(Evaluation):
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Entered the calculate score method of the RMSE class")
            rmse=np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info(f"The root mean squared error valuee is {rmse}")
            return rmse
        except Exception as e:
            logging.error(e)
            raise e
        