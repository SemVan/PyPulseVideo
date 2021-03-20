import csv
import os
import numpy as np



CONTACTLESS_NAME = "color.txt"
CONTACT_NAME = "Contact.txt"
NORMAL_LENGTH = 500

def read_contact_file(fileName):
    data = []
    with open(fileName, 'r') as f:
        for row in f.read().splitlines():
            rowList = row.split("/n")
            for elem in rowList:
                try:
                    data.append(float(elem))
                except:
                    continue

    return np.asarray(data)


def read_contactless_file(fileName):
    # if os.path.isfile(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            rowList = row.split(" ")
            data[0].append(rowList[0])
            if len(rowList)==2:
                data[1].append(float(rowList[1]))
            else:
                if (float(rowList[0])+float(rowList[2])>0):
                    data[1].append(float(rowList[1])/(float(rowList[0])+float(rowList[2])))
                else:
                    data[1].append(float(rowList[1]))
    return np.asarray(data[1])


def get_dir_list(path):
    dir_list = []
    for filder in os.listdir(path):
        n = path + '/' + filder + '/'
        dir_list.append(n)
    return dir_list


def get_signals_dict(dir_list, contact_name):
    data = []
    total_broken = 0
    for d in dir_list:
        print(d)
        dname = d.split('/')[-1]
        nd = d.replace("MAHNOB_VIDEOS", 'ECG')
        contact_file = nd + contact_name
        less_file = d + CONTACTLESS_NAME

        contact_signal = read_contact_file(contact_file)
        less_signal = read_contactless_file(less_file)

        lens = []
        lens.append(len(less_signal))
        if len(contact_signal) >= NORMAL_LENGTH and len(less_signal) >= NORMAL_LENGTH:
            data.append([contact_signal[0:NORMAL_LENGTH-1], less_signal[0:NORMAL_LENGTH-1]])
    print("TOTAL BROKEN ", total_broken)
    print(np.min(lens), np.max(lens))
    return data


def write_csv_dataset(dataset, filename):
    with open (filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in dataset:
            writer.writerow(row[0])
            writer.writerow(row[1])
    return




def main():
    normal_list = get_dir_list("./mahnob/MAHNOB_ECG/mahnob/MAHNOB_VIDEOS")
    # metro_list = []
    # metro_list += get_dir_list("./Metrological/Distances")
    # metro_list += get_dir_list("./Metrological/Intensity")
    # full_data = []
    full_data = get_signals_dict(normal_list, CONTACT_NAME)
    input(full_data)
    # full_data += get_signals_dict(metro_list, "Contactless.txt")
    print(len(full_data))
    write_csv_dataset(full_data, "dataset_mahnob.csv")


main()
