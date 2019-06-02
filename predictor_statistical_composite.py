import math
import pickle
from datetime import timedelta

import pandas as pd
from predictor import Predictor
from predictor_control_statistical import PredictorControlStatistical

def resample_and_extend(data, freq):
    data = data.resample(freq).mean()
    half = int(round(len(data) / 2))
    after = data[:half].shift(len(data), freq=freq)
    before = data[half:].shift(-len(data), freq=freq)
    return pd.concat([before, data, after])

def cyclical_moving_average(data, freq, window):
    extended = resample_and_extend(data, freq)
    return extended.rolling(window, center=True, win_type="triang").mean()[data.index[0]:data.index[len(
        data.index) - 1]]  # Choice of win_type could be more informed. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html, https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows

class PredictorStatisticalComposite(Predictor):
    def __init__(self, subset, period):
        self.helper = PredictorControlStatistical(subset, period)
        self.subset = subset
        self.period = period

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))

    def train(self, params, combined, load, households):
        baseline_load = (combined-load).mean(axis="columns")
        seasonal = cyclical_moving_average(baseline_load, "d", 30)
        hourly = baseline_load.groupby(baseline_load.index.hour).mean()
        seasonal = seasonal.append(pd.Series({seasonal.index[-1]+timedelta(days=1): float("NaN")}))    #Need to add one more day to get all the hours of the last day
        composite = seasonal.resample("h").ffill()
        for time in composite.index:
            composite.loc[time] = (composite.loc[time]+hourly.loc[time.hour])/2
        composite = composite.resample(str(self.period)+"min").ffill().iloc[:-1]    #Resample to our desired frequency and get rid of the extra day we added
        self.baselines = composite

        self.helper.train(params, combined.sub(self.baselines, axis="index"), load, households)    #May have to modify later if PredictorControlStatistical.train gets more complicated
        self.helper.l2_threshold = 3482   #It seems like this threshold should be lower than the one in predictor_control_statistical, so I'm not sure why tuning for the n_L1/n_L2 ratio gets it to be higher.

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    def predict(self, params, combined):
        return self.helper.predict(params, combined.sub(self.baselines, axis="index"))