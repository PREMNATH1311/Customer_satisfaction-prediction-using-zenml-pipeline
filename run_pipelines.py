from pipelines.training_pipelines import training_pipeline
from zenml.client import Client
if __name__ =="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="C:/ml projects/znmlprojects/testing project/data/olist_customers_dataset.csv")
    
# mlflow ui --backend-store-uri "file:C:\Users\unkno\AppData\Roaming\zenml\local_stores\47a5f68f-d468-4b4b-a21a-fc6a7c4f9a5e\mlruns"