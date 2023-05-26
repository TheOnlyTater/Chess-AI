import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from scipy.signal import savgol_filter
import scipy.stats as st

data_dict = {}
sortedDict = {}
plt.style.use('bmh')

# Give a string of the data filename or location of the file(only for csv)
dataFileLocation = ["neuralNetResults/conv/base.csv"]

# Set the index of each item you want to be graphed in the dataframe in an array
# desieredGraphsArray = [
#    (0, 2, "Games Played", "Move Confidence",
#     "Move Confidence Derived", "dxMoveConf", "dx"),
#    (0, 4, "Games Played", "Pieces Captured",
#     "Pieces Captured Derived", "dxMovePiece", "dx"),
#    (0, 5, "Games Played", "AVG inv", "AVG Inv derived", "dxAVGinv", "dx"),
#    (0, 4, "Games Played", "Pieces Caputred",

#     "Pieces Captured filtered", "filtPieceCap", "filt"),
#    (0, 3, "Games Played", "Total Game Moves",
#     "Total Moves Derived", "dxGameMoves", "dx"),
#    (0, 1, "Games Played", "Result", "Results Derived", "dxResult", "dx"),
#    (0, 5, "Games Played", "AVG inv", "AVG Inv (filtered)", "filtAvgInv", "filt"),
#    (0, 3, "Games Played", "Total Game Moves",
#     "Total Moves Filtered", "filtTotalMoves", "filt"),
#    (0, 2, "Games Played", "Move Confidence",
#     "Move Confidence Filtered", "filtMoveConf", "filt"),
#    (0, 2, "Move Confidence", "Times Occured",
#        "Move Confidence distrobution", "histMoveConf", "hist"),
#    (0, 1, "Games Played", "Result", "Game Outcome", "dfltResult", "dflt"),
#    (0, 2, "Games Played", "Move Confidence",
#     "Move Confidence", "dfltMoveConf", "dflt"),
#    (0, 5, "Games Played", "AVG Inv", "Average Invalid", "dfltavginv", "dflt")
# ]

# desieredGraphsArray = [
#    (4, "conf"),
#    (4, "devi"),
#    (2, "conf"),
#    (2, "devi"),
#    (3, "conf"),
#    (3, "devi"),
# ]

desieredGraphsArray = [
    (0, 4, "Pieces Captured", "Times Occured", "piecehist", "hist")
]

desgrahs = [0, 1, 2, 3, 4, 5]

possibleavgs = ["med", "var", "cmn", "avginv", "devi", "conf"]
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


def deviation(samples):
    return np.std(samples)


def confidenceIntervals(data):
    return st.norm.interval(alpha=0.99,
                            loc=np.mean(data),
                            scale=st.sem(data)
                            )


def graphInclination(data):
    return 0


def estimateDerivative(y_valueList, acc=0.0001):
    derivatives = []
    try:
        for x, y in enumerate(y_valueList):
            new_x = (2*x + 1) / 2
            new_y = (y + y_valueList[x + 1]) / 2
            while (y + acc) < new_y:
                new_x = (new_x + x) / 2
                new_y = (new_y + y) / 2
            dy = (new_y - y) / (new_x - x)
            derivatives.append(dy)

    except IndexError:
        return derivatives
# Different plot functions


def defaultPlot(x, y):
    plt.plot(x, y)


def scatterPlot(ax, x, y):
    ax.scatter(x, y, s=[10**2.5*n for n in y])


def errorBarPlot(ax, x, y, yerr):
    ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)


def histPlot(ax, x):
    ax.bar(list(x.keys()), x.values())

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
    elif desieredPlot == "devi":
        return deviation(sortedDict[graphIndex])
    elif desieredPlot == "conf":
        return confidenceIntervals(sortedDict[graphIndex])


def choosePlot(ax, desieredPlot, graphinfo):
    if desieredPlot == "dflt":
        defaultPlot(graphinfo[0], graphinfo[1])
    elif desieredPlot == "dx":
        y = estimateDerivative(graphinfo[1])
        defaultPlot(graphinfo[0][:-1], y)
    elif desieredPlot == "err":
        err = func(graphinfo[1])
        errorBarPlot(ax, graphinfo[0], graphinfo[1], err)
    elif desieredPlot == "sct":
        scatterPlot(ax, graphinfo[0], graphinfo[1])
    elif desieredPlot == "hist":
        val = commonNumber(graphinfo[1])
        histPlot(ax, val)
    elif desieredPlot == "filt":
        y = savgol_filter(graphinfo[1], 77, 7)
        defaultPlot(graphinfo[0], y)
    elif desieredPlot == "filterr":
        y = savgol_filter(graphinfo[1], 77, 7)
        err = func(y)
        errorBarPlot(ax, graphinfo[0], y, err)


def main(dataFrames, valsFromData, desGraphs, combineGraphs=None):
    dfArr = []

    for df in dataFrames:
        df = pd.read_csv(df, skiprows=1, names=["Game Result"])
        dfArr.append(df)
    csvToData(dfArr)

    for file, dsGraph in enumerate(valsFromData):
        vals = sortDesiredData(data_dict, dsGraph)
        sortedDict[file] = vals

    fig, ax = plt.subplots()
    for graph in desGraphs:
        if graph[-1] in possibleavgs:
            avg = chooseAvg(graph[0], graph[-1])
            print(avg)
        else:
            x, y = sortedDict[graph[0]], sortedDict[graph[1]]
            xName, yName, title = graph[2], graph[3], graph[4]
            filename = graph[5]
            choosePlot(ax, graph[-1], [x, y])
            plt.xlabel(xName)
            plt.ylabel(yName)
            plt.title(title)
            plt.savefig(filename, dpi=100)
            plt.clf()
        print(f"{graph[-1]} is done")


main(dataFileLocation, desgrahs, desieredGraphsArray)

# desieredGraphsArray = [
#    (0, 2, "Games Played", "Move Confidence", "Move Confidence Scatter Graph", "scatMoveConf", "sct")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)
# desieredGraphsArray = [(0, 3, "Total Game Moves", "Times Occured",
#                        "Total Game Moves distribution", "histMoveTotal", "hist")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)

# desieredGraphsArray = [
#    (0, 2, "Games Played", "Move Confidence", "filterrMoveConf", "filterr")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)

# desieredGraphsArray = [
#    (0, 2, "Move Confidence", "Times Occured", "Move Confidence distribution(10k Games)", "histMoveConf", "hist")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)

# desieredGraphsArray = [
#    (0, 4, "Pieces Captured", "Times Occured", "Pieces Captured distribution(10k Games)", "histPieceCap", "hist")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)

# desieredGraphsArray = [
#    (0, 2, "Move Confidence", "Times Occured",
#     "Move Confidence distribution", "histMoveConf", "hist")]
# main(dataFileLocation, desgrahs, desieredGraphsArray)
