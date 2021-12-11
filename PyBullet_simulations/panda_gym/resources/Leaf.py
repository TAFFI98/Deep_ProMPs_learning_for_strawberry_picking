import os
import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import pybullet as p
from suppress_warning import suppress_stdout
class Leaf:
    '''
    Loads the URDF of the Leaf in the simulation
    '''
    def __init__(self, client_id, start_pose, start_ori,Short=True,Medium=False,Long=False):
        if Short:
            self.leaf_urdf_folder = currentdir + "/urdf/leaf/leaf_short.urdf"
        elif Medium:
            self.leaf_urdf_folder = currentdir + "/urdf/leaf/leaf_medium.urdf"
        elif Long:
            self.leaf_urdf_folder = currentdir + "/urdf/leaf/leaf_long.urdf"
        with suppress_stdout():
         self.leaf = p.loadURDF(fileName =self.leaf_urdf_folder,
                               basePosition = start_pose,
                               baseOrientation = start_ori,
                               useFixedBase = True,
                               physicsClientId = client_id)

