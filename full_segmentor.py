import os
from video_processor import *
import time
from file_writer import *
from segmentor import *

# VIDEO_PATH = "./Videos/Measurements/"
# FILES_PATH = "./Videos/Segmented/"
VIDEO_PATH = "./mahnob/MAHNOB_VIDEOS/"
FILES_PATH = "./mahnob/MAHNOB_ECG/"

for filename in os.listdir(VIDEO_PATH):
    if filename.endswith(".avi"):
        name = filename[:-4]
        full_path = VIDEO_PATH + filename
        #text_path = VIDEO_PATH + "/Signals/" + name
        text_path = FILES_PATH + name
        file_name = text_path + "/" + name + ".csv"
        if not os.path.isdir(text_path):
            os.makedirs(text_path)
        print (text_path)
        start = time.time()
        seg_sig = get_segmented_video(full_path)
        print(time.time() - start)
        if seg_sig is not None and len(seg_sig) != 0    :
            write_segmented_file(file_name, seg_sig)
