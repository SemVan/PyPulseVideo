import os
from video_processor import *
import time
from file_writer import *

VIDEO_PATH = "./Videos"
FILES_PATH = "./Signals"

for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        full_path = VIDEO_PATH + "/" + filename
        text_path = VIDEO_PATH + "/Signals/" + name
        if not os.path.isdir(text_path):
            os.makedirs(text_path)
        print (text_path)
        start = time.time()
        geom, color = full_video_file_procedure(full_path)
        print(time.time() - start)
        if not geom == None:
            geom_path = text_path +'/'+"geom.txt"
            color_path = text_path +'/'+"color.txt"
            write_file(geom_path, geom)
            write_file(color_path, color)
