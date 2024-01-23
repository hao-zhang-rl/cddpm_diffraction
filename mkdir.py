import os
import shutil
class Checkifexists():
    def __init__(self,New_folder):
        if os.path.exists(New_folder) and os.path.isdir(New_folder):
            shutil.rmtree(New_folder)
        try:
            os.makedirs(New_folder)
        except OSError:
            print("Creation of the main directory '%s' failed " % New_folder)
        else:
            print("Successfully created the main directory '%s' " % New_folder)