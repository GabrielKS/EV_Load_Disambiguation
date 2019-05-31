from abc import ABC, abstractmethod


class Predictor(ABC):
    trained = False

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass    #Should return a Predictor loaded from the file at path

    @abstractmethod
    def train(self, params, combined, load, households):
        trained = True

    @abstractmethod
    def predict(self, params, combined):
        assert self.trained, "Predictor is not trained."
        # Should return {"load": loadPrediction, "households": householdPrediction}

    @abstractmethod
    def save(self, path):
        assert self.trained, "Predictor is not trained."    #May want to modify this behavior