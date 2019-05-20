import pandas as pd
import numpy as np

def print_summary(data):
    print("Data:")
    print(data)
    print("calculating averages")
    averages = data.mean(axis="index")
    print("Averages:")
    print(averages)
    print("Overall average: "+str(np.mean(averages)))

vehicles_to_households = pd.read_csv("raw_data/vehicles.csv")
print(vehicles_to_households)
v = [int(s[8:]) for s in vehicles_to_households["Vehicle"]]
h = [int(s[10:]) for s in vehicles_to_households["Household"]]
print(v)
print(h)
print([i in h for i in range(1, 201)] == [True]*200)    #Everyone has a car
counts = [h.count(i) for i in range(1, 201)]
print("Number of households with 1 car: "+str(counts.count(1)))
print("Number of households with 2 car: "+str(counts.count(2)))
print("Number of households with 3 cars: "+str(counts.count(3)))
print("Number of households with 4 cars: "+str(counts.count(4)))
print("Number of households with 5 cars: "+str(counts.count(5)))
print("Number of households with 6 cars: "+str(counts.count(6)))
print("Number of households with 7 cars: "+str(counts.count(7)))

baseline = pd.read_csv("raw_data/Residential-Profiles.csv")
load_L1 = pd.read_csv("raw_data/PEV-Profiles-L1.csv")
load_L2 = pd.read_csv("raw_data/PEV-Profiles-L2.csv")
print_summary(baseline)
print_summary(load_L1)
print_summary(load_L2)
print(load_L1.loc[1704])