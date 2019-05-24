from predictor import Predictor
import pandas as pd
import numpy as np
import utilities

class PredictorControlStatistical(Predictor):
    def __init__(self, subset, period):
        self.subset = subset
        self.period = period
        self.watts_per_car = (12000/1)*(30/100)*(1/8760)*(1000/1)    #(mi/yr)(kWh/mi)(yr/h)(W/kW)=W. Sources: https://www.immihelp.com/used-car-buying-tips/, https://en.wikipedia.org/wiki/Electric_car_energy_efficiency
        self.blocks_per_day = self.watts_per_car*1440/period    #This number is highly suspect. Next step would be to get it from the training data.

    def load(self, path):
        pass

    def train(self, params, combined, load, households):
        pass

    def save(self):
        pass

    def isL2(self, data, charging_threshold):
        #L2 should indicate spikier data, which should lead to fewer datapoints above charging_threshold
        l2_threshold = 20   #Should be tuned/trained
        # print(len(data[data > charging_threshold]))
        return len(data[data > charging_threshold]) > l2_threshold

    def predict(self, params, combined):
        cars_L1 = params["vehicles_L1"]*self.subset
        cars_L2 = params["vehicles_L2"]*self.subset
        cars_total = params["vehicles_total"]*self.subset
        households_total = len(combined.columns)
        cars_per_household = cars_total/households_total
        fraction_EV = (cars_L1+cars_L2)/cars_total

        #1 car: 34%; 2 cars: 31%; 3+ cars: 35% (source: https://www.reference.com/world-view/many-cars-average-american-family-own-f0e6dffd882f2857, but I doubt it)
        fraction_one_car = .40  #A guess.
        fraction_two_cars = 3-2*fraction_one_car-cars_per_household #To make the math work
        fraction_three_cars = 1-fraction_one_car-fraction_two_cars  #To make the math work

        zero_EVs, one_EV, two_EVs, three_EVs = 0, 0, 0, 0

        #Surely there's a way to do this in fewer lines, but (see notebook):
        three_EVs += fraction_three_cars*fraction_EV**3
        two_EVs += fraction_three_cars*(fraction_EV**2*(1-fraction_EV))*3
        one_EV += fraction_three_cars*(fraction_EV*(1-fraction_EV)**2)*3
        zero_EVs += fraction_three_cars*(1-fraction_EV)**3

        two_EVs += fraction_two_cars*fraction_EV**2
        one_EV += fraction_two_cars*(fraction_EV*(1-fraction_EV))*2
        zero_EVs += fraction_two_cars*(1-fraction_EV)**2

        one_EV += fraction_one_car*fraction_EV
        zero_EVs += fraction_one_car*(1-fraction_EV)

        # print(zero_EVs+one_EV+two_EVs+three_EVs)    #Should add to 1

        zero_EVs, one_EV, two_EVs, three_EVs = int(round(households_total*zero_EVs)), int(round(households_total*one_EV)), int(round(households_total*two_EVs)), int(round(households_total*three_EVs)) #Surely there's a better way; this seems very non-Pythonic.

        # print((one_EV+2*two_EVs+3*three_EVs))   #Should equal cars_L1+cars_L2 up to the rounding
        # print(str(cars_L1+cars_L2)+"/"+str(cars_total))

        watts_per_household = combined.mean(axis="rows")
        watts_per_household.sort_values(inplace=True)
        households = pd.DataFrame(float(0), index=combined.columns, columns=["L1", "L2"])
        load = pd.DataFrame(float(0), index=combined.index, columns=combined.columns)

        for index, row in households.iterrows():
            cars = 0
            if watts_per_household.index.get_loc(index) > zero_EVs: cars += 1
            if watts_per_household.index.get_loc(index) > zero_EVs+one_EV: cars += 1
            if watts_per_household.index.get_loc(index) > zero_EVs+one_EV+two_EVs: cars += 1
            for i in range(0, cars):
                charging_thresholds = {}
                blocks_above = 0
                for day in pd.date_range(start="2010-01-01 00:00:00", end="2010-01-08 00:00:00", freq="D"):
                    today = combined[index].loc[day:day+pd.DateOffset(days=1, minutes=-self.period)]
                    car_consumption = self.blocks_per_day*cars
                    charging_thresholds[day] = utilities.threshold_for_sum_above(today, car_consumption)
                    blocks_above += list(today > charging_thresholds[day]).count(True)
                is_L2 = blocks_above/cars > 125 #Tune/train it! Depends on the time period too. Maybe use the threshold utility function.
                if is_L2: households.at[index, "L2"] += 1
                else: households.at[index, "L1"] += 1
                power = 6600 if is_L2 else 1920
                for day in pd.date_range(start="2010-01-01 00:00:00", end="2010-01-08 00:00:00", freq="D"):
                    today = combined[index].loc[day:day+pd.DateOffset(days=1, minutes=-self.period)]
                    sorted = today.sort_values(ascending=False)
                    car_consumption = self.blocks_per_day*cars
                    j = 0
                    while car_consumption > 0:
                        load.at[sorted.index[j], index] = power #We have our answer for this timeslot
                        combined.at[sorted.index[j], index] -= power #Decrease the power available to the next car
                        car_consumption -= power    #Decrease the power needed to charge this car
                        j += 1  #Move to the next-highest timeslot
        return {"load": load, "households": households}