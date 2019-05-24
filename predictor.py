from abc import ABC, abstractmethod


class Predictor(ABC):
    trained = False

    @abstractmethod
    def load(self, path):
        trained = True  #May want to modify this behavior

    @abstractmethod
    def train(self, params, combined, load, households):
        trained = True

    @abstractmethod
    def predict(self, params, combined):
        assert self.trained, "Predictor is not trained."
        # Should return {"load": loadPrediction, "households": householdPrediction}

    @abstractmethod
    def save(self):
        assert self.trained, "Predictor is not trained."    #May want to modify this behavior