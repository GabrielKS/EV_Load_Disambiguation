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
            n = 5   #I'd need to do more research to figure out what this should be. Too low and it takes forever and then just gives every multiple of 5 up to the maximum index; too high and it only gives the last index. In the middle, it produces a short list of numbers for the first household (and maybe a small minority of the others?) and only the last index for the rest. I haven't found a number that produces *something* but not *everything* for all households.
            print(n)
            stdev = combined[household].std()
            print("\tpredicting")
            x = detector.predict(epsilon=3*n*stdev**2)  #I'm not sure what to put in for the parameter here. Options are epsilon (don't konw what that is), "pen" for penalty (don't know how to calculate that), and "n_bkps" for the number of breakpoints (which is unknown).
            print(x)
        households = pd.DataFrame(0, index=combined.columns, columns=["L1", "L2"])
        return {"load": load, "households": households}