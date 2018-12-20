import os
import shutil


VIDEO_PATH = "./Videos/Measurements/"
FILES_PATH = "./Videos/Signals/"
OLD_FILES_PATH = "./Videos/Old_signals/"

for foldername in os.listdir(FILES_PATH):
    print(foldername)
    contact_dir = OLD_FILES_PATH+ foldername
    if os.path.isdir(contact_dir):
        print(contact_dir)
        contact_file = contact_dir + "/" + "Contact.txt"
        src_dir = FILES_PATH + foldername
        if os.path.isfile(contact_file):
            shutil.copy(contact_file, src_dir)
