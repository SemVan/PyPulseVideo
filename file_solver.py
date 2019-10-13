import os


VIDEO_PATH = "./Measurements/"
FILES_PATH = "./Segmented/Signals/"
FILE_NAME = "signal.csv"

for folder in os.listdir(FILES_PATH):
    file_lister = os.listdir(FILES_PATH+folder)
    if 'signal.csv' not in file_lister:
        print(folder)
