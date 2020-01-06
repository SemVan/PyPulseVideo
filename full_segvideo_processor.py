import os
from video_processor import *
import time
from file_writer import *
from segmented_io import *
from segmentor import *



VIDEO_PATH = "./Metrological/Int_videos/"
FILES_PATH = "./Metrological/Intensity/"
FILE_NAME = "signal_int.csv"

loglist = ["ab", "cd"]
for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        try:
            name = filename[:-4]
            full_path = VIDEO_PATH + filename
            text_path = FILES_PATH + name
            #file_name = text_path + "/" + name + ".csv"
            file_name = text_path + "/" + FILE_NAME
            if not os.path.isdir(text_path):
                os.makedirs(text_path)
            print (text_path)
            start = time.time()
            seg_sig = get_segmented_video(full_path)
            print(time.time() - start)
            if not len(seg_sig) == 0:
                write_segmented_file(file_name, seg_sig)
            loglist.append(full_path)
        except:
            print("blya epta")
            continue

with open("seglogger_intensity.txt", 'w') as f:
    for fil in loglist:
        f.write(fil+"\n")
