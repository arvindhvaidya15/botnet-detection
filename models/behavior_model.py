from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

class BehaviorModel:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.1)

    def train(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = sum(predictions == y) / len(y)
        return accuracy
