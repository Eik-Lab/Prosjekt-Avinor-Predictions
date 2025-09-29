from src.train import TrainConfig, train, model_predict
from src.model import create_XGBoost

def main():
    data_path = "data/historical_flights.csv"


    config = TrainConfig(flight_csv = data_path, 
                         model_name = "xg", 
                         model_function = create_XGBoost)
    model = train(config)

    model_predict("data/schedule_oct2025.csv", model)
    

if __name__ == "__main__":
    main()