import numpy as np
import json
from matplotlib import pyplot as plt
import scipy.stats


def read_metrics(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


def parse_name(name):
    str_lst = name.split('/')
    n = str_lst[-2]
    n = n.split('_')[-2]
    return n

def parse_distance_name(name):
    input(name)
    str_lst = name.split('/')
    n = str_lst[-2]
    n = n.split('_')[-2]
    return n

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h, h

def parce_one_type(d):
    result_map = {}
    for elem in d:
        dude = list(elem.keys())[0]
        if not ("gadzhimirzaev" in dude):
            param = parse_name(dude)
            if not param in result_map:
                result_map[param] = []
            result_map[param].append(100 * elem[list(elem.keys())[0]]['less'])
    return result_map


def parce_net_name(d):
    result_map = {}
    for elem in d:
        print(elem)
        dude = elem["dir"] + '/'
        if not ("gadzhimirzaev" in dude):
            param = parse_name(dude)
            if not param in result_map:
                result_map[param] = []
            result_map[param].append(100 * elem['res'])
    return result_map

def get_t_test(d1, d2):
    return scipy.stats.ttest_ind(d1, d2)

def get_full_t_test(data1, data2):
    ks = [int(k) for k in data1.keys()]
    ks = sorted(ks)
    for param in ks:
        print(ks, get_t_test(data1[str(param)], data2[str(param)]))
    input()
    return

def get_median(data):
    dsort = sorted(data)
    return dsort[int(len(dsort)/2)]

def prepare(data, label):
    x = []
    y = []
    y_low = []
    y_high = []
    ks = [int(k) for k in data.keys()]
    ks = sorted(ks)
    for param in ks:
        # print(param)
        # input(data[str(param)])
        m, ml, mh = mean_confidence_interval(data[str(param)])
        x.append(float(param)/ 100)
        y.append(np.mean(data[str(param)]))
        # y.append(get_median(data[str(param)]))
        y_low.append(ml)
        y_high.append(mh)

    plt.plot(x, y, label=label)
    # plt.errorbar(x, y, yerr=(y_low, y_high), label=label)
    # plt.scatter(x, y)

# intense = read_metrics("intensity.json")
dist = read_metrics("intensity_1_less.json")
input(dist)
# dist_net = read_metrics("intensity_net_1.json")

# input(intense)
# intense = parce_one_type(intense)
dist = parce_one_type(dist)
# dist_net = parce_net_name(dist_net)
# get_full_t_test(dist_net, dist)

# prepare(dist, "Old algorithm")
prepare(dist, "New algorithm")
plt.xlabel("Освещенность БО, м")
plt.ylabel("Q-метрика, %")
plt.ylim([0, 7])
# plt.legend(loc='upper left')
plt.grid(True)
plt.show()
