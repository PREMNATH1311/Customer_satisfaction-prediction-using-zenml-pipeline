import logging
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self) -> None:
        pass
        
    def get_data(self)->pd.DataFrame:
        #logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv("data/olist_customers_dataset.csv")
    
@step
def ingest_df() -> pd.DataFrame:
    try:
        ingest_data=IngestData()
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data:{e}")
        raise e
        
