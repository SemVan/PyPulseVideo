import os
from video_processor import *
import time
from file_writer import *

VIDEO_PATH = "./Videos/Measurements/"
FILES_PATH = "./Videos/Signals/"

for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        full_path = VIDEO_PATH + filename
        #text_path = VIDEO_PATH + "/Signals/" + name
        text_path = FILES_PATH + name
        if not os.path.isdir(text_path):
            os.makedirs(text_path)
        print (text_path)
        start = time.time()
        seg_sig = get_segmented_video(text_path)
        print(time.time() - start)
        if not seg_sig == None:
            write_segmented_file(sef_path, seg_sig)
