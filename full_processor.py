import os
from video_processor import *
import time
from file_writer import *

VIDEO_PATH = "./Metrological/Dist_videos/"
FILES_PATH = "./Metrological/Distances/"
ALL_FILES = set(["colgeom.txt", 'geom.txt', 'color.txt'])

def get_default_list(direct):
    dirlst = []
    namelist = []
    for filename in os.listdir(direct):
        full_path = VIDEO_PATH + filename
        dirlst.append(full_path)
        namelist.append(filename.split('.')[0])
    return dirlst, namelist


def get_what_to_redo(direct, lst, dirlst):
    redo_list = []
    for subdir in os.listdir(direct):
        if subdir in lst:
            fuldir = direct + '/' + subdir
            print(fuldir)
            files_in_dir = []
            for filename in os.listdir(fuldir):
                files_in_dir.append(filename)
            if not(ALL_FILES <= set(files_in_dir)):
                redo_list.append(subdir)
        else:
            redo_list.append(subdir)
    redo_dirlist = []
    for filename in redo_list:
        for dirname in dirlst:
            if filename in dirname:
                redo_dirlist.append(dirname)
    input(redo_dirlist)
    return


dr, fl = get_default_list(VIDEO_PATH)
get_what_to_redo(FILES_PATH, fl, dr)
input()
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
