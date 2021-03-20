from matplotlib import pyplot as plt
from scipy.signal import medfilt

import os
import numpy as np
from classification_math import *
FILES = ['RR_color.txt','RR_geom.txt','RR_colgeom.txt']
ECG_file = 'ECG.txt'

NAMES = ['ECG','color','geom','colgeom']

IMENA = ['ЭКГ','Цветовой','Геометрический','Цветогеометрический']

def read_file(fileName):
    data = []
    with open(fileName, 'r') as f:
        for row in f:
            row = float(row.split("\n")[0])
            data.append(row)
    return np.asarray(data)

def inds(gbplf,gbplf2):
    folders = os.listdir('./Metrological/Intensity')
    SInds = [[],[],[],[]]
    for folder in folders:
        f = './Metrological/Intensity' + '/' + folder + '/'
        files = os.listdir(f)
        if ((ECG_file in files) & (FILES[0] in files)):
            RRc = read_file(f+FILES[0])
            RRg = read_file(f+FILES[1])
            RRcg = read_file(f+FILES[2])
            RRe = read_file(f+ECG_file)/25
            
            RR = [RRe,RRc,RRcg,RRcg]
            
#             fig, axs = plt.subplots(2, 2)
            for i in range(len(NAMES)):
                if i>0:
                    RR[i] = RR[i][RR[i]>0.5]
                    RR[i] = RR[i][RR[i]<2]
    #                 print(len(RR[i]))
#                     gbplf = 11
                    mask = np.ones(gbplf)/gbplf
    #                 print(mask)
                    RR[i] = np.convolve(RR[i],mask,'valid')
                    RR[i] = medfilt(RR[i],gbplf2)
    #             print(len(RR[i]))
                n = i//2
                m = i%2
#                 axs[m,n].scatter(RR[i][0:-1],RR[i][1:])
#                 axs[m,n].set_title(NAMES[i])
    #             axs[m,n].set(xlim=(0.75,1.25),ylim=(0.75,1.25))
                W = np.max(RR[i])-np.min(RR[i])
                h, edg = np.histogram(RR[i],bins=5)
                edg = (edg[:-1] + edg[1:])/2
                x = np.argmax(h)
                M = h[x]
                AMo = M / np.sum(h)
                Mo = edg[x]
    #             SInd = str(AMo / (2*Mo*W))
                SInd = '%.2f' % (AMo / (2*Mo*W))
#                 axs[m,n].set_title(IMENA[i] + ', $\mathregular{И_н}$ = ' + SInd)
#                 axs[m,n].set_xlabel('$\mathregular{RR_{i+1}}$, с')
#                 axs[m,n].set_ylabel('$\mathregular{RR_i}$, с')
#                 axs[m,n].grid()
                SInd = AMo / (2*Mo*W)
                SInds[i].append(SInd)
                # axs[m,n].bar(edg,h)
    #         plt.show()
                    
                    
# print(SInds)

#     stats = []

    return np.asarray(SInds)

# for Inds in SInds:
#     m = np.mean(Inds)
#     s = np.std(Inds) / len(Inds) ** 0.5
#     stats.append([m,s,len(Inds)])
    
# # print(stats)

# ref = stats[0]
# stats = stats[1:]

# for s in stats:
#     t = np.abs(s[0]-ref[0])/s[1]
#     print(t)
#     print(s[2])
#     print("===========")
        
    
Min = 1000000
for i in range(12):
    for j in range(12):
        res = inds(i+1,2*j+1)
        res = res[3]-res[0]
        res = np.sum(res**2)
#         print(res)
        if res<Min:
            Min = res
            I = i
            J = j
            
print(I)
print(J)

print(np.transpose(inds(I+1,2*J+1),(1,0)))