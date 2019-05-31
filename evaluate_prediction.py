import get_input
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import predictor_control_statistical
import predictor_statistical_composite
import predictor_statistical_daily
import predictor_zero

split = .25

n_vehicles = len(pd.read_csv("raw_data/vehicles.csv"))
n_L1 = n_vehicles * 0.75 * 0.3
n_L2 = n_vehicles * 0.75 * 0.6
d_L1 = n_L1*0.05
d_L2 = n_L2*0.05
timestep = 3

def get_train_test(split):
    seed = 1000
    random.seed(a=seed)
    np.random.seed(seed)
    get_input.verbose = True
    train = get_input.get_data(n_L1, d_L1, n_L2, d_L2, timestep, 0)
    test = get_input.get_data(n_L1, d_L1, n_L2, d_L2, timestep, 1)
    get_input.verbose = False

    households = train["households"]["Household"]
    training_households, testing_households = train_test_split(households, test_size=split)

    train["combined"] = train["combined"][list(training_households)]
    train["load"] = train["load"][list(training_households)]
    train["households"] = train["households"].loc[train["households"]["Household"].isin(training_households)]

    test["combined"] = test["combined"][list(testing_households)]
    test["load"] = test["load"][list(testing_households)]
    test["households"] = test["households"].loc[test["households"]["Household"].isin(testing_households)]
    return train, test

def rmse_2d(a, b):
    return ((b-a) ** 2).mean().mean() ** 0.5

def evaluate_prediction(predictor, data):
    print("Results for "+type(predictor).__name__+":")
    data = data.copy()
    data["households"] = data["households"].copy(deep=True)
    prediction = predictor.predict(data["params"], data["combined"])

    data["households"].index = (data["households"])["Household"]
    data["households"].drop("Household", axis=1, inplace=True)

    load_RMSE = rmse_2d(data["load"], prediction["load"])
    households_RMSE = rmse_2d(data["households"], prediction["households"])
    print("\tLoad RMSE: "+str(round(load_RMSE, 2)))
    print("\tHousehold RMSE: "+str(round(households_RMSE, 3)))

def create_all_predictors():
    # return [predictor_statistical_composite.PredictorStatisticalComposite(1-split, 30)]
    return [predictor_zero.PredictorZero(), predictor_statistical_composite.PredictorStatisticalComposite(1-split, 30), predictor_statistical_daily.PredictorStatisticalDaily(1-split, 30), predictor_control_statistical.PredictorControlStatistical(1-split, 30)]

def testSaveLoad(predictors):
    for predictor in predictors:
        filename = type(predictor).__name__+"_"+get_input.path_from(n_L1, d_L1, timestep, 0)+".csv"
        predictor.save(filename)
        predictor = predictor.load(filename)

def main():
    train, test = get_train_test(split)
    predictors = create_all_predictors()
    print("training…")
    for predictor in predictors:
        predictor.train(train["params"], train["combined"], train["load"], train["households"])
    print("saving/loading…")
    testSaveLoad(predictors)
    print("evaluating…")
    for predictor in predictors:
        evaluate_prediction(predictor, train)

if __name__ == "__main__":
    main()