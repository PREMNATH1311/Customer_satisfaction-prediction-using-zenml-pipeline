import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataPreProcessStratgy,DataCleaning,DataDivideStrategy,DataStrategy
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_df(df:pd.DataFrame)-> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        preprocess_strategy=DataPreProcessStratgy()
        data_cleaning=DataCleaning(df,preprocess_strategy)
        preeprocessed_data=data_cleaning.handle_data()
        
        divide_strategy=DataDivideStrategy()
        data_cleaning=DataCleaning(preeprocessed_data,divide_strategy)
        x_train,x_test,y_train,y_test=data_cleaning.handle_data()
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logging.error(e)
        raise e