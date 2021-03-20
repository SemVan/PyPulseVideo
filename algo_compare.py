import json
from matplotlib import pyplot as plt
import numpy as np

FILES_TO_READ = ["instensity_net_1.json", "distance_net_1.json", "distances_1_all_algo.json", "intensity_1_all_algo.json"]
ALGO_NAMES = {'geom': "Landmarks based", 'color': "RGB based", 'colgeom': "Combined", 'net': "Landmarks net based"}
INT_LABELS = ["Bioobject illuminance, lx", "Q-metric"]
DIST_LABELS = ["Distance to bioobjet, cm", "Q-metric"]
MARKERS = ["o", "v", "s", "P"]

def read_json(filename):
    with open (filename) as f:
        data = json.load(f)
    return data


def parce_intensity(filename):
    param = filename.split('_')[-2]
    return param

def parce_distance(filename):
    param = filename.split('_')[-2]
    return param

def sort_data(result):
    for algo in result:
        X = result[algo][0]
        Y = result[algo][1]
        X, Y = zip(*sorted(zip(X,Y)))
        result[algo] = [X, Y]
    return result

def build(full_data):
    plots_int = {}
    for param in full_data:
        algo_mean = {}
        for algo_set in full_data[param]:
            for algo in algo_set:
                if not algo in algo_mean:
                    algo_mean[algo] = []
                if not algo in plots_int:
                    plots_int[algo] = [[],[]]
                algo_mean[algo].append(algo_set[algo])
        for algo in algo_mean:
            plots_int[algo][1].append(np.mean(algo_mean[algo]))
            plots_int[algo][0].append(int(param))
    return sort_data(plots_int)

def plot_data(data, xlabel, ylabel):
    plt.figure()
    i = 0
    for algo in plots_int:
        plt.plot(data[algo][0], data[algo][1], marker=MARKERS[i], label=ALGO_NAMES[algo])
        plt.grid(True)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fancybox=True, shadow=False, ncol=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        i += 1
    plt.show()


full_data_intensity = {}
full_data_distance = {}
for filename in FILES_TO_READ:
    data = read_json(filename)
    for elem in data:
        measure_file = list(elem.keys())[0]
        if "Intensity" in measure_file:
            param = parce_intensity(measure_file)
            if param not in full_data_intensity:
                full_data_intensity[param] = []
            full_data_intensity[param].append(elem[measure_file])
        if "Distances" in measure_file:
            param = parce_distance(measure_file)
            if param not in full_data_distance:
                full_data_distance[param] = []
            full_data_distance[param].append(elem[measure_file])

input(full_data_distance)
plots_int = build(full_data_intensity)
plots_dist = build(full_data_distance)

plot_data(plots_int, INT_LABELS[0], INT_LABELS[1])
plot_data(plots_dist, DIST_LABELS[0], DIST_LABELS[1])
