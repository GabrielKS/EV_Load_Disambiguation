import numpy as np
import pandas as pd

def sum_above(data, x):
    return np.sum([d-x for d in data[data > x]])

def threshold_for_sum_above(data, sum):  #Finds the number x such that sum([d-x for d in data[data > x]])=sum
    if len(data) <= 0:
        # raise ValueError("data must be non-empty")
        return float("NaN")
    data = np.array(data)
    data.sort()
    i_low = len(data)
    b = 0
    for i in range(0, len(data)):   #Iterate across the data from low to high
        b = sum_above(data, data[i])    #b is the sum above a given point
        # print("b: "+str(b))
        if b <= sum:    #The first time that b fits within sum, stop iterating. We know that x is somewhere in (data[i-1], data[i]]
            i_low = i-1
            break
    if i_low >= len(data):
        # raise ValueError("sum exceeds the sum of the data")
        return float("NaN")
    s_low = data[i_low] if i_low >=0 else -float("inf")
    s_high = data[i_low+1]
    n = len(data[data>s_low])
    x = s_high-(sum-b)/n
    assert np.isclose(sum_above(data, x), sum), "incorrect threshold, x="+str(x)+", sum="+str(sum)+", sum_above="+str(sum_above(data, x))
    return x

def main():
    print(threshold_for_sum_above([1, 2, 3], 1.5))
    print(threshold_for_sum_above(pd.Series({2: 1, 3: 2, 4: 3}), 1.5))
    print(threshold_for_sum_above(pd.Series({2: 1, 3: 2, 4: 3}).index, 1.5))

if __name__ == "__main__":
    main()