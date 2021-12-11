import numpy as np
import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import pybullet as p
from suppress_warning import suppress_stdout
class Strawberry:
    '''
    Loads the URDF of the red or green strawberry in the simulation in a certain random position
    '''
    def __init__(self,client_id, start_pose, start_ori, red, ShortMedLong = [1,0,0]):

        if red == False and ShortMedLong == [1,0,0]:
            self.file_name = currentdir + "/urdf/green_strawberry/berry_short.urdf"
        elif red == False and ShortMedLong == [0,1,0]:
            self.file_name = currentdir + "/urdf/green_strawberry/berry_medium.urdf"
        elif red == False and ShortMedLong == [0,0,1]:
            self.file_name = currentdir + "/urdf/green_strawberry/berry_long.urdf"
        elif red == True and ShortMedLong == [1,0,0]:
            self.file_name = currentdir + "/urdf/red_strawberry/berry_short.urdf"
        elif red == True and ShortMedLong == [0,1,0]:
            self.file_name = currentdir + "/urdf/red_strawberry/berry_medium.urdf"
        elif red == True and ShortMedLong == [0,0,1]:
            self.file_name = currentdir + "/urdf/red_strawberry/berry_long.urdf"
        # Load URDF
        with suppress_stdout():
          self.strawberry = p.loadURDF(fileName = self.file_name,
                                    basePosition = start_pose,
                                    baseOrientation = start_ori,
                                    useFixedBase = True,
                                    physicsClientId = client_id)


    def get_strawberry_position(self,client_id):
        # Get the current position of the strawberry
        strawberry_position = []
        strawberry_state = p.getLinkState(self.strawberry, 0, computeForwardKinematics=True,physicsClientId=client_id)
        for i in range(3):
            strawberry_position.append(strawberry_state[0][i])
        strawberry_position = np.array(strawberry_position)
        return strawberry_position

