import os
from video_processor import *
import time
from file_writer import *
<<<<<<< HEAD:full_segmentor.py
from segmentor import *

VIDEO_PATH = "./Videos/Measurements/"
FILES_PATH = "./Videos/Segmented/"
=======
from segmented_io import *


VIDEO_PATH = "./Videos/Measurements/"
FILES_PATH = "./Segmented/Signals/"
>>>>>>> eb6a1cbd8e2c105c4d2771b47e1db65180df0826:full_segvideo_processor.py

for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        full_path = VIDEO_PATH + filename
        text_path = FILES_PATH + name
        file_name = text_path + "/" + name + ".csv"
        if not os.path.isdir(text_path):
            os.makedirs(text_path)
        print (text_path)
        start = time.time()
        seg_sig = get_segmented_video(full_path)
        print(time.time() - start)
        if not seg_sig == None:
            write_segmented_file(file_name, seg_sig)
