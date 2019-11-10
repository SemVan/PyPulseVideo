import os
from video_processor import *
import time
from file_writer import *

VIDEO_PATH = "./Metrological/Int_videos/"
FILES_PATH = "./Metrological/Intensity/"

what_to_read = []
for filename in os.listdir(VIDEO_PATH):
    print(filename)
input()


for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        print(name)
        full_path = VIDEO_PATH + filename
        # #text_path = VIDEO_PATH + "/Signals/" + name
        text_path = FILES_PATH + name
        if not os.path.isdir(text_path):
            os.makedirs(text_path)
        print (text_path)
        start = time.time()
        try:
            geom, color, colgeom = full_video_file_procedure(full_path)
            print(time.time() - start)
            if not geom == None:
                geom_path = text_path +'/'+"geom.txt"
                color_path = text_path +'/'+"color.txt"
                colgeom_path = text_path +'/'+"colgeom.txt"
                write_file(geom_path, geom)
                write_file(color_path, color)
                write_file(colgeom_path, colgeom)
                what_to_read.append(text_path +'/' + '\n')
        except:
            print("eba eba blyat!" + full_path)
with open("logger.txt", 'w') as f:
    for line in what_to_read:
        f.write(line)
