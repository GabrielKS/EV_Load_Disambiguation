import pickle

import pandas as pd
from predictor import Predictor
from predictor_control_statistical import PredictorControlStatistical


class PredictorStatisticalDaily(Predictor):
    def __init__(self, subset, period):
        self.helper = PredictorControlStatistical(subset, period)
        self.subset = subset
        self.period = period

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))

    def train(self, params, combined, load, households):
        baseline_load = combined-load
        self.baselines = baseline_load.mean(axis="columns")
        self.helper.train(params, combined.add(self.baselines, axis="index"), load, households)    #May have to modify later if PredictorControlStatistical.train gets more complicated
        self.helper.l2_threshold = 84   #It seems like this threshold should be lower than the one in predictor_control_statistical, so I'm not sure why tuning for the n_L1/n_L2 ratio gets it to be higher.

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    def predict(self, params, combined):
        return self.helper.predict(params, combined.sub(self.baselines, axis="index"))