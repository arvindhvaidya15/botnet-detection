from utils.preprocess import load_data, preprocess_data
from models.behavior_model import BehaviorModel
import pandas as pd

def main():
    data = load_data('data/network_traffic.csv')
    data = preprocess_data(data)

    features = data[['duration', 'packets', 'bytes']]
    labels = data['label']

    model = BehaviorModel()
    model.train(features)
    accuracy = model.evaluate(features, labels)

    print(f'Model Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
