import matplotlib.pyplot as plt
import pandas as pd
import evaluate_prediction

f = 1

def resample_and_extend(data, freq):
    data = data.resample(freq).mean()
    half = int(round(len(data)/2))
    after = data[:half].shift(len(data), freq=freq)
    before = data[half:].shift(-len(data), freq=freq)
    return pd.concat([before, data, after])

def cyclical_moving_average(data, freq, window):
    extended = resample_and_extend(data, freq)
    return extended.rolling(window, center=True, win_type="triang").mean()[data.index[0]:data.index[len(data.index)-1]] #Choice of win_type could be more informed. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html, https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows

def averages():
    global f
    train, test = evaluate_prediction.get_train_test(.25)
    combined = train["combined"].mean(axis="columns")

    #Seasonal trends:
    daily = combined.resample("d").mean()
    seasonal = cyclical_moving_average(combined, "d", 14)
    plt.figure(f)
    f += 1
    plt.plot(daily.index, daily, seasonal.index, seasonal)

    #Hourly trends:
    hourly = combined.groupby(combined.index.hour).mean()
    plt.figure(f)
    f += 1
    plt.plot(hourly.index, hourly)

    #Composite:
    composite = seasonal.resample("h").ffill()
    for time in composite.index:
        composite.loc[time] = (composite.loc[time]+hourly.loc[time.hour])/2
    plt.figure(f)
    f += 1
    plt.plot(combined.index, combined, composite.index, composite)

    #TODO: Trends in each hour across seasons

def household_with_EVs():
    global f
    train, test = evaluate_prediction.get_train_test(.25)
    combined = train["combined"]["Household 23"][:500]
    load = train["load"]["Household 23"][:500]
    plt.figure(f)
    f += 1
    plt.plot(combined.index, combined, load.index, load)
    plt.axhline(y=1920)
    plt.axhline(y=6600)

def differences():
    global f
    train, test = evaluate_prediction.get_train_test(.25)
    combined = train["combined"]["Household 23"][:500]
    load = train["load"]["Household 23"][:500]
    differences1 = combined-combined.shift(1)
    differences2 = combined-combined.shift(2)
    differences3 = (combined+combined.shift(-1))/2-(combined.shift(1)+combined.shift(2))/2
    differences4 = (combined+combined.shift(-1))/2-(combined.shift(2)+combined.shift(3))/2
    plt.figure(f)
    f += 1
    # plt.plot(combined.index, combined, load.index, load)
    plt.plot(differences1.index, differences1, differences2.index, differences2, differences3.index, differences3, differences4.index, differences4)
    plt.axhline(y=1920)
    plt.axhline(y=-1920)
    plt.axhline(y=6600)
    plt.axhline(y=-6600)

def main():
    # averages()
    household_with_EVs()
    differences()
    plt.show()

if __name__ == "__main__":
    main()