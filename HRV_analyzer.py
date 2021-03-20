import os
import numpy as np
from matplotlib import pyplot as plt

COLOR_FILE = "RR_color.txt"
GEOM_FILE = "RR_geom.txt"
COLGEOM_FILE = "RR_colgeom.txt"
CONTACT_FILE = "RR_Contact.txt"
ECG_FILE = "ECG.txt"
KEY_LIST = ["color", "geom", "colgeom", "ECG", "contact"]
FILE_NAME_MAP = {"color": COLOR_FILE, "geom": GEOM_FILE, "colgeom": COLGEOM_FILE, "contact": CONTACT_FILE, "ECG": ECG_FILE}
SIGNAL_MAP = {"color": "color.txt", "geom": "geom.txt", "colgeom": "colgeom.txt", "contact": "Contact.txt", "ECG": "ECG.txt"}
PARAMS_LIST = ["SDNN", "RMSSD", "NN50", "pNN50", "KVar", "D", "As", "Ex", "Mo", "AMo", "MxDMn", "IN", "C1", "C0", "L", "w", "S"]
HISTOGRAM_BINS = np.linspace(400, 1300, num = 18)


def prepare_no_log_dir_list():
    dir_list = []
    for folder in os.listdir("./Metrological/Intensity/"):
        n = "./Metrological/Intensity/" + folder + '/'
        dir_list.append(n)
    return dir_list

def write_file(file_name, metrics):
    with open(file_name,'w') as txtfile:
        for key in PARAMS_LIST:
            row = key + ": " + str(metrics[key]) + '\n'
            txtfile.write(row)
    return

def params_processor():
    dir_list = prepare_no_log_dir_list()
    print("Directory listing done")
    metrics_map = {}
    for key in KEY_LIST:
        metrics_map[key] = []
    metrics = {}
    for dir in dir_list:
        try:
            for key in KEY_LIST:
                file_dir = dir + FILE_NAME_MAP[key]
                RRs = read_file(file_dir)
                if (key == "ECG"):
                    RRs = RRs / 25
                RRs = RRs * 1000  # convert to milliseconds
                
                # statistic methods
                
                metrics["SDNN"] = np.std(RRs)
                metrics["RMSSD"] = ( np.sum(np.diff(RRs) ** 2) / (len(RRs) - 1) ) ** 0.5
                metrics["NN50"] = np.sum(np.abs(np.diff(RRs)) > 50)
                metrics["pNN50"] = metrics["NN50"] / (len(RRs) - 1)
                metrics["KVar"] = metrics["SDNN"] / np.mean(RRs)
                metrics["D"] = metrics["SDNN"] ** 2
                metrics["As"] = np.sum((RRs - np.mean(RRs)) ** 3) / (len(RRs) * metrics["SDNN"] ** 3)
                metrics["Ex"] = np.sum((RRs - np.mean(RRs)) ** 4) / (len(RRs) * metrics["SDNN"] ** 4) - 3
                
                # histogram method
                
                hist = np.histogram(RRs, bins = HISTOGRAM_BINS)
                h = hist[0][hist[0] > 0]
                metrics["MxDMn"] = np.max(RRs) - np.min(RRs)
                metrics["AMo"] = np.max(h)
                metrics["Mo"] = hist[1][np.argmax(hist[0])]
                metrics["IN"] = metrics["AMo"] / (2 * metrics["Mo"] * metrics["MxDMn"])
    #             plt.plot(hist[1], hist[0])
    #             plt.show()
                
                # correlation
                if (key != "ECG"):
                    sig = normalize(read_signal_file(dir + SIGNAL_MAP[key]))
                    corr = np.correlate(sig, sig, "full")
                    metrics["C1"] = corr[len(corr) // 2 - 1]
                    i = len(corr) // 2
                    c = corr[i]
                    while (c > 0):
                        i = i - 1
                        c = corr[i]
                
                    metrics["C0"] = (len(corr) // 2 - i) * 40  # milliseconds
                else:
                    metrics["C1"] = 0
                    metrics["C0"] = 0
                
                # scatterogram method
                
                RRn = RRs[0: -2]
                RRn1 = RRs[1: -1]
    #             plt.plot(RRn, RRn1)
    #             plt.show()
                RRl = np.matmul([[0.7, 0.7],[-0.7, 0.7]], [RRn, RRn1])
                RRl1 = RRl[1]
                RRl = RRl[0]
                metrics["L"] = np.max(RRl) - np.min(RRl)
                metrics["w"] = np.max(RRl) - np.min(RRl)
                metrics["S"] = 3.14 * metrics["w"] * metrics["L"]
                    
            
                write_file(dir + key + "_HRV.txt", metrics)
    #             metrics_map[key].append(metrics)
                # вообще стоит сохранить файл с метриками в папке!
        except:
            print(dir)
    # общие выводы по показателям
    return 0

def read_file(fileName):
    data = []
    with open(fileName, 'r') as f:
        for row in f:
            data.append(float(row))
    return np.asarray(data)

def get_y_reverse_signal(sig):
    sig_max = np.max(sig)
    new_sig = sig_max-sig
    return new_sig


def read_signal_file(fileName):
    # if os.path.isfile(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(" ")
            data[0].append(rowList[0])
            if len(rowList) < 3:
                rowList = rowList[0].split(",")
                data[1].append(float(rowList[1]))
            else:
                if (float(rowList[0])+float(rowList[2])>0):
                    data[1].append(float(rowList[1])/(float(rowList[0])+float(rowList[2])))
                else:
                    data[1].append(float(rowList[1]))
    return get_y_reverse_signal(np.asarray(data[1]))


def normalize(sig):
    sig = sig - np.mean(sig)
    sig = sig / np.sum(sig ** 2) ** 0.5
    return sig

out = params_processor()