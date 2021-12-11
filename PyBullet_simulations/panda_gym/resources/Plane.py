import os
import pybullet_data
import pybullet as p
class Plane:
    '''
    Loads the URDF of the Plane in the simulation from pybullet_data
    '''
    def __init__(self,client_id):
        self.plane = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),physicsClientId=client_id)
