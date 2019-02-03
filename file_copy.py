import os
import shutil


VIDEO_PATH = "./Videos/Measurements/"
FILES_PATH = "./Videos/Signals_geom_and_color_color/"
OLD_FILES_PATH = "./Videos/Old_signals/"

for foldername in os.listdir(FILES_PATH):
    print()

    contact_dir = OLD_FILES_PATH+ foldername
    if os.path.isdir(contact_dir):
        print(contact_dir)
        contact_file = contact_dir + "/" + "Contact.txt"
        src_dir = FILES_PATH + foldername
        print ("from" + contact_file)
        print("to" + src_dir)
        if os.path.isfile(contact_file):
            if not os.path.isfile(src_dir+"/Contact.txt"):
                shutil.move(contact_file, src_dir)
