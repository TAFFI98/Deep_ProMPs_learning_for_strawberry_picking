import numpy as np
import os,inspect
import pybullet as p
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from suppress_warning import suppress_stdout

class Table:
    '''
    Loads the URDF of the Table in the simulation in a certain position
    '''
    def __init__(self,table_position,client_id):
        self.table_urdf_folder= currentdir +"/urdf/table/table.urdf"
        self.table_position= np.array(table_position)
        with suppress_stdout():
         self.table = p.loadURDF(self.table_urdf_folder, self.table_position, physicsClientId=client_id)


