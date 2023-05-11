import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil

data_dict = {}
sortedDict = {}
plt.style.use('bmh')

# Give a string of the data filename or location of the file(only for csv)
dataFileLocation = ["base.csv"]

# Set the index of each item you want to be graphed in the dataframe in an array
desieredGraphsArray = [(0, 1, "test", "test2", "check", "dflt")]

desgrahs = [0, 3]

possibleavgs = ["med", "var", "cmn", "avginv"]
# Gather data from csv functions


def csvToData(DFs):
    for count, DF in enumerate(DFs):
        data_dict[count] = []
        for ind, row in DF.iterrows():
            data = [float(i) for i in str(row["Game Result"]).split(";")]
            data_dict[count].append(data)


def sortDesiredData(data, col):
    lst = []
    for key in data:
        for i in data[key]:
            lst.append(i[col])
    return lst

# different functions of how to gather and display data


def func(data):
    arr = np.array(data)
    avg = np.sum(arr) / len(data)
    return np.sqrt(np.sum((arr - avg)**2) / (len(data) - 1))


def median(data):
    data = np.sort(data)
    if len(data) % 2 == 0:
        return (data[len(data) / 2] + data[(len(data) / 2)+1]) / 2
    return data[ceil(len(data) / 2)]


def variaton(data):
    return data[-1] - data[0]


def commonNumber(data):
    return {i: data.count(i) for i in data}


def estimateDerivative(y_valueList, acc=0.0001):
    derivatives = []
    try:
        for x, y in enumerate(y_valueList):
            new_x = (2*x + 1) / 2
            new_y = (y + y_valueList[x + 1]) / 2
            while (y + acc) > new_y:
                new_x = (new_x + x) / 2
                new_y = (new_y + y) / 2
            dy = (new_y - y) / (new_x - x)
            derivatives.append((x, dy))

    except KeyError:
        return derivatives
# Different plot functions


def defaultPlot(x, y, xAxis: str, yAxis: str):
    plt.plot(x, y)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)


def scatterPlot(ax, x, y, size, color, vmin=0, vmax=100):
    ax.scatterPlot(x, y, s=size, c=color, vmin=vmin, vmax=vmax)


def errorBarPlot(ax, x, y, yerr):
    ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)


def histPlot(ax, x):
    ax.hist(x, bins=8, linewidth=0.5, edgecolor='white')


# function where every graph is created
def chooseAvg(graphIndex, desieredPlot: str):
    if desieredPlot == "med":
        return median(sortedDict[graphIndex])
    elif desieredPlot == "var":
        return variaton(sortedDict[graphIndex])
    elif desieredPlot == "cmn":
        return commonNumber(sortedDict[graphIndex])
    elif desieredPlot == "avginv":
        return func(sortedDict[graphIndex])


def choosePlot(ax, desieredPlot, graphinfo):
    if desieredPlot == "dflt":
        defaultPlot(graphinfo[0], graphinfo[1], graphinfo[2], graphinfo[3])


def main(dataFrames, valsFromData, desGraphs, combineGraphs=None):
    dfArr = []

    for df in dataFrames:
        df = pd.read_csv(df, skiprows=1, names=["Game Result"])
        dfArr.append(df)
    csvToData(dfArr)

    for file, dsGraph in enumerate(valsFromData):
        vals = sortDesiredData(data_dict, dsGraph)
        sortedDict[file] = vals

    ax = plt.subplots()
    for graph in desGraphs:
        if graph[-1] in possibleavgs:
            avg = chooseAvg(graph[0], graph[-1])
        else:
            x, y = sortedDict[graph[0]], sortedDict[graph[1]]
            xName, yName = graph[2], graph[3]
            filename = graph[4]
            choosePlot(ax, graph[-1], [x, y, xName, yName])

            plt.show()
            # plt.savefig(filename)


main(dataFileLocation, desgrahs, desieredGraphsArray)
