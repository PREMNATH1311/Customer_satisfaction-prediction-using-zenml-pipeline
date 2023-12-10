import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
class DataStrategy(ABC):
    
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->pd.DataFrame:
        pass
    
class DataPreProcessStratgy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data=data.drop(["order_approved_at",
                            "order_delivered_carrier_date",
                            "order_delivered_customer_date",
                            "order_estimated_delivery_date",
                            "order_purchase_timestamp"],
                           axis=1)
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)
            
            data=data.select_dtypes(include=[np.number])
            cols_to_drop=["customer_zip_code_prefix","order_item_id"]
            data=data.drop(cols_to_drop,axis=1)
            return data
        except Exception as e:
            logging.error(e)
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            x=data.drop("review_score",axis=1)
            y=data["review_score"]
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error(e)
            raise e
        
class DataCleaning:
    def __init__(self,data:pd.DataFrame,strategy=DataStrategy) -> None:
        self.df=data
        self.strategy=strategy
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(e)
            raise e
        
if __name__ == "__main__":
    data=pd.read_csv("C:/ml projects/znmlprojects/testing project/data/olist_customers_dataset.csv")
    data_cleaning=DataCleaning(data,DataPreProcessStratgy())
    data_cleaning.handle_data()
    s=DataDivideStrategy()
    s.handle_data(data)
