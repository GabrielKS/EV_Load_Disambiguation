import pandas as pd
from predictor import Predictor
import ruptures as rpt
import numpy as np

class PredictorChangePoint(Predictor):
    def __init__(self, search, cost):
        # self.search = rpt.Binseg
        # self.cost = "l1"
        self.search = search
        self.cost = cost

    @classmethod
    def load(cls, path):
        pass

    def train(self, params, combined, load, households):
        pass

    def save(self, path):
        pass

    def predict(self, params, combined):
        load = pd.DataFrame(index=combined.index, columns=combined.columns)
        for household in combined:
            print(household)
            detector = self.search(model=self.cost)
            detector.fit(np.array(combined[household]))
            # n = len(combined[household])/1000
            n = 5
            print(n)
            stdev = combined[household].std()
            print("\tpredicting")
            x = detector.predict(epsilon=3*n*stdev**2)
            print(x)
        households = pd.DataFrame(0, index=combined.columns, columns=["L1", "L2"])
        return {"load": load, "households": households}