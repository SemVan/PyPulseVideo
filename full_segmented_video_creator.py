import os
from video_processor import *
import time
from file_writer import *
from segmented_io import *
from segvideo_creator import *



VIDEO_PATH = "./Videos/Measurements/"
EDITED_PATH = "./Edited/"
FILES_PATH = "./Segmented/Signals/"
FILE_NAME = "flag.csv"


dir_list = []


for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        if name == "Andreeva_1_27.04.19":
            filename_edited = name + "_edited" + ".avi"
            full_path = VIDEO_PATH + filename
            full_path_edited = EDITED_PATH + filename_edited
            metric_path = FILES_PATH + name
            metric_name = metric_path + "/" + FILE_NAME
            if not os.path.isdir(metric_path):
                os.makedirs(metric_path)
            print (metric_path)
            start = time.time()
            make_edited_video(full_path, full_path_edited, metric_path)
            print(time.time() - start)
            break
