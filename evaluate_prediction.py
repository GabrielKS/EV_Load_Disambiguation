import get_input
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import predictor_control_statistical

def get_train_test(split):
    seed = 1000
    random.seed(a=seed)
    np.random.seed(seed)

    n_vehicles = len(pd.read_csv("raw_data/vehicles.csv"))
    n_L1 = n_vehicles * 0.75 * 0.3
    n_L2 = n_vehicles * 0.75 * 0.6
    d_L1 = n_L1*0.05
    d_L2 = n_L2*0.05
    train = get_input.get_data(n_L1, d_L1, n_L2, d_L2, 3, 0)
    test = get_input.get_data(n_L1, d_L1, n_L2, d_L2, 3, 1)

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

def main():
    split = .25
    train, test = get_train_test(split)
    predictor = predictor_control_statistical.PredictorControlStatistical(split, 30)
    predictor.train(train["params"], train["combined"], train["load"], train["households"])
    prediction = predictor.predict(test["params"], test["combined"])

    test["households"].index = (test["households"])["Household"]
    test["households"].drop("Household", axis=1, inplace=True)

    load_RMSE = rmse_2d(test["load"], prediction["load"])
    households_RMSE = rmse_2d(test["households"], prediction["households"])
    print(load_RMSE)
    print(households_RMSE)

if __name__ == "__main__":
    main()