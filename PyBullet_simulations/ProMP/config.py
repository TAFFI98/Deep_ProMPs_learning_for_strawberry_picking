import os,inspect

''' FOLDERS DEFINITNION '''

ROOT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # Root directory --->  ..../ProMP
PARENT_DIR = os.path.dirname(ROOT_DIR)                                               # Parent directory --->  ..../RtP_Clusuter

''' Deep models paths '''
''' 0 '''
MODEL_FOLDER_0 = os.path.join(ROOT_DIR, "deep models/J0")

''' 1 '''
MODEL_FOLDER_1 = os.path.join(ROOT_DIR, "deep models/J1")

''' 2 '''
MODEL_FOLDER_2 = os.path.join(ROOT_DIR, "deep models/J2")

''' 3 '''
MODEL_FOLDER_3 = os.path.join(ROOT_DIR, "deep models/J3")

''' 4 '''
MODEL_FOLDER_4 = os.path.join(ROOT_DIR, "deep models/J4")

''' 5 '''
MODEL_FOLDER_5 = os.path.join(ROOT_DIR, "deep models/J5")

''' 6 '''
MODEL_FOLDER_6 = os.path.join(ROOT_DIR, "deep models/J6")

''' f1 '''
MODEL_FOLDER_f1 = os.path.join(ROOT_DIR, "deep models/f1")


''' f2 '''
MODEL_FOLDER_f2 = os.path.join(ROOT_DIR, "deep models/f2")
