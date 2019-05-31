from abc import ABC, abstractmethod


class Predictor(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path):
        pass    #Should return a Predictor loaded from the file at path

    @abstractmethod
    def train(self, params, combined, load, households):
        pass    #Should train the Predictor

    @abstractmethod
    def predict(self, params, combined):
        pass    # Should return {"load": loadPrediction, "households": householdPrediction}

    @abstractmethod
    def save(self, path):
        pass    #Should save the Predictor to a file at path