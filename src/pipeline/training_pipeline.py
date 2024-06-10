from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    data_ingestion = DataIngestion()
    data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)