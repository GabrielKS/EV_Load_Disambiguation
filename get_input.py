import pandas as pd
import numpy as np
import random
from collections import OrderedDict

def load_time_series(path):
    print("loading")
    var = pd.read_csv(path)
    var.index = pd.to_datetime(var["Time"])
    var.drop(columns="Time", inplace=True)
    return var

def get_input(vehicles_L1, vehicles_L2, timestep, random_seed=None):
    if random_seed == None: return simulate_input(vehicles_L1, vehicles_L2, timestep)
    path = "input/load_"+str(timestep)+"_"+str(vehicles_L1)+"_"+str(vehicles_L2)+"_"+str(random_seed)+".csv"
    try:
        return load_time_series(path)
    except FileNotFoundError:
        input, _, _ = simulate_input(vehicles_L1, vehicles_L2, timestep)
        input.to_csv(path)
        return input

def load_resample(path, key, distribution, freq, vehicles_to_households):
    var = load_time_series(path)
    var = var[vehicles_to_households["Vehicle"].iloc[distribution == key]] #Get rid of the data we don't need
    print("adding resolution")
    for s in var:
        print("\t"+s)
        col = var[s].copy()
        for i in range(0, len(col)-1):
            if col.iloc[i] == 0 and col.iloc[i+1] > 0:
                max = col.iloc[i+1]
                offset = random.random()-.5
                if (offset < 0): col.iloc[i] = max*-offset
                else: col.iloc[i+1] = max*(1-offset)
                # col.iloc[i] = -1
            if col.iloc[i] > 0 and col.iloc[i+1] == 0:
                max = col.iloc[i]
                offset = random.random()-.5
                if (offset < 0): col.iloc[i] = max*(1+offset)
                else: col.iloc[i+1] = max*offset
                # col.iloc[i] = -2
        var[s] = col
    #var = var.resample(freq).mean()
    print(var)
    return var

def simulate_input(vehicles_L1, vehicles_L2, timestep, random_seed=None):
    print("Simulating input from scratch…")
    vehicles_to_households = pd.read_csv("raw_data/vehicles.csv")
    distribution = np.array(["L1"] * vehicles_L1 + ["L2"] * vehicles_L2 + ["Gas"] * (len(vehicles_to_households) - vehicles_L1 - vehicles_L2))
    random.seed(a=random_seed)
    random.shuffle(distribution)

    baseline = load_time_series("raw_data/Residential-Profiles.csv")
    freq = (baseline.index[1]-baseline.index[0])*timestep
    baseline = baseline.resample(freq).mean()

    load_L1 = None
    load_L2 = None

    if vehicles_L1 > 0: load_L1 = load_resample("raw_data/PEV-Profiles-L1.csv", "L1", distribution, freq, vehicles_to_households)
    if vehicles_L2 > 0: load_L2 = load_resample("raw_data/PEV-Profiles-L2.csv", "L2", distribution, freq, vehicles_to_households)

    print("calculating")
    """
    for household_n in range(1, len(baseline+1)):
        household = "Household "+str(household_n)
        vehicle_ns = vehicles_to_households[vehicles_to_households["Household"] == household].index
        for vehicle_n in vehicle_ns:
            print("\t"+str(vehicle_n))
            if distribution[vehicle_n] == "L1": baseline[household] = baseline[household]+load_L1.iloc[vehicle_n]
            elif distribution[vehicle_n] == "L2": baseline[household] = baseline[household]+load_L2.iloc[vehicle_n]
    """
    #Instead of for every household iterating over its vehicles, let's iterate over all vehicles and pick households:
    load = pd.DataFrame(0, index=baseline.index, columns=baseline.columns)
    households_to_vehicles = pd.DataFrame({"Household": list(OrderedDict.fromkeys(vehicles_to_households["Household"])), "L1": 0, "L2": 0}, columns=["Household", "L1", "L2"])
    for index, row in vehicles_to_households.iterrows():
        vehicle = row["Vehicle"]
        household = row["Household"]
        if distribution[index] == "L1" or distribution[index] == "L2":
            households_to_vehicles.loc[households_to_vehicles["Household"] == household, distribution[index]] += 1
            if distribution[index] == "L1": load[household] += load_L1[vehicle]
            else: load[household] += load_L2[vehicle]

    return baseline+load, load, households_to_vehicles

def main():
    #Overall EV market share: https://evadoption.com/ev-market-share/; let's say 25% like in Palo Alto
    #Can't find level 1 vs level 2 stats beyond the fact that L2 is much more convenient; assuming 30% L1 and 60% L2
    #30-minute intervals for load data is on the optimistic end of plausible
    #Using random seed of 0 for reproducibility
    n_vehicles = len(pd.read_csv("raw_data/vehicles.csv"))
    n_L1 = int(n_vehicles * 0.75 * 0.3)
    n_L2 = int(n_vehicles * 0.75 * 0.6)
    print(simulate_input(n_L1, n_L2, 3, 0))

if __name__ == "__main__":
    main()