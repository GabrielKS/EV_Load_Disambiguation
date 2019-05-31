import pandas as pd
from predictor import Predictor

class PredictorZero(Predictor):
    def __init__(self):
        pass

    @classmethod
    def load(cls, path):
        return PredictorZero()  #No state to load, so we just return a new one

    def train(self, params, combined, load, households):
        pass

    def save(self, path):
        pass    #No state to save

    def predict(self, params, combined):
        load = pd.DataFrame(float(0), index=combined.index, columns=combined.columns)
        households = pd.DataFrame(0, index=combined.columns, columns=["L1", "L2"])
        return {"load": load, "households": households}