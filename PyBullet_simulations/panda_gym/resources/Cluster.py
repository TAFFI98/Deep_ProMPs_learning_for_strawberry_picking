import pybullet as p
import numpy as np
import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)
import Leaf
import Strawberry
from suppress_warning import suppress_stdout

class Cluster:
    def __init__(self, cluster_pose, client_id):
        with suppress_stdout():
         support_pose =cluster_pose
         support_pose[0] -= 0.05
         self.support = p.loadURDF(fileName = currentdir + "/urdf/support/support.urdf",basePosition = support_pose,baseOrientation = p.getQuaternionFromEuler([0, 0, 1.57]),useFixedBase = True,physicsClientId = client_id)
        ''' Randomize n of strawberries, leaves and length of branches'''
        rand = np.random.random_integers(low=0, high=10)
        if rand >= 3 and rand <= 7 :
          self.n_short_red_berries = 1  #  np.random.random_integers(low=2, high=10)
          self.n_medium_red_berries = 0# np.random.random_integers(low=2, high=10)
          self.n_long_red_berries = 0 # np.random.random_integers(low=2, high=10)
        elif rand < 3:
          self.n_short_red_berries = 0#  np.random.random_integers(low=2, high=10)
          self.n_medium_red_berries =1 # np.random.random_integers(low=2, high=10)
          self.n_long_red_berries = 0 # np.random.random_integers(low=2, high=10)
        elif rand > 7:
          self.n_short_red_berries = 0 #  np.random.random_integers(low=2, high=10)
          self.n_medium_red_berries = 0 # np.random.random_integers(low=2, high=10)
          self.n_long_red_berries = 1 # np.random.random_integers(low=2, high=10)

        self.n_medium_green_berries =   np.random.random_integers(low=3, high=4)
        self.n_short_green_berries =   np.random.random_integers(low=3, high=4)
        self.n_long_green_berries =  np.random.random_integers(low=3, high=4)

        self.n_leaves_short  =np.random.random_integers(low=4, high=5)
        self.n_leaves_medium = np.random.random_integers(low=4, high=5)
        self.n_leaves_long = np.random.random_integers(low=3, high=4)

        ''' Upload ripe strawberries'''

        for i in range(self.n_long_red_berries):
            berry_pose = [1.19, 1.0, 1.44]
            # Noise on position
            # Noise on y position
            r = np.random.random_integers(low=0, high=9)
            if r == 0 or r == 1:
                y_noise = -0.13
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 2 or r == 3:
                y_noise = 0.13
                self.app_angle_min = -0.15
                self.app_angle_max = 0.35
            elif r == 4 or r == 5:
                y_noise = 0
                self.app_angle_min = -0.1
                self.app_angle_max = 0.4
            elif r == 6 or r == 7:
                y_noise = -0.05
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 8 or r == 9:
                y_noise = 0.05
                self.app_angle_min =  -0.1
                self.app_angle_max = 0.45
            x_noise = np.random.uniform(low=0.0, high=0.005)
            berry_pose[1] += y_noise
            berry_pose[0] -= x_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            self.red_strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or,
                                                        red=True, ShortMedLong=[0, 0, 1])



            for i in range(0, p.getNumJoints(self.red_strawberry.strawberry)):
                p.changeDynamics(self.red_strawberry.strawberry, i, lateralFriction=0, spinningFriction=0,
                                 rollingFriction=0, restitution=0, contactStiffness=0.0, contactDamping=0.00000,
                                 linearDamping=0.0000, angularDamping=0.00000)

        for i in range(self.n_short_red_berries):
            berry_pose = [1.19, 1.0, 1.44]
            # Noise on y position
            r = np.random.random_integers(low=0, high=9)
            if r == 0 or r == 1:
                y_noise = -0.13
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 2 or r == 3:
                y_noise = 0.13
                self.app_angle_min = -0.15
                self.app_angle_max = 0.35
            elif r == 4 or r == 5:
                y_noise = 0
                self.app_angle_min =-0.1
                self.app_angle_max = 0.4
            elif r == 6 or r == 7:
                y_noise = -0.05
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 8 or r == 9:
                y_noise = 0.05
                self.app_angle_min =  -0.1
                self.app_angle_max = 0.45

            x_noise = np.random.uniform(low=0.0, high=0.005)
            berry_pose[1] += y_noise
            berry_pose[0] -= x_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            self.red_strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or,
                                                        red=True, ShortMedLong=[1, 0, 0])

            for i in range(0, p.getNumJoints(self.red_strawberry.strawberry)):
                p.changeDynamics(self.red_strawberry.strawberry, i, lateralFriction=0, spinningFriction=0,
                                 rollingFriction=0, restitution=0, contactStiffness=0.0, contactDamping=0.00000,
                                 linearDamping=0.0000, angularDamping=0.00000)

        for i in range(self.n_medium_red_berries):
            berry_pose = [1.19, 1.0, 1.44]
            # Noise on position
            # Noise on y position
            r = np.random.random_integers(low=0, high=9)
            if r == 0 or r == 1:
                y_noise = -0.13
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 2 or r == 3:
                y_noise = 0.13
                self.app_angle_min = -0.15
                self.app_angle_max = 0.35
            elif r == 4 or r == 5:
                y_noise = 0
                self.app_angle_min = -0.1
                self.app_angle_max = 0.4
            elif r == 6 or r == 7:
                y_noise = -0.05
                self.app_angle_min = -0.1
                self.app_angle_max = 0.5
            elif r == 8 or r == 9:
                y_noise = 0.05
                self.app_angle_min =  0.1
                self.app_angle_max = 0.45
            x_noise = np.random.uniform(low=0.0, high=0.005)
            berry_pose[1] += y_noise
            berry_pose[0] -= x_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            self.red_strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or,
                                                        red=True, ShortMedLong=[0, 1, 0])


            for i in range(0, p.getNumJoints(self.red_strawberry.strawberry)):
                p.changeDynamics(self.red_strawberry.strawberry, i, lateralFriction=0, spinningFriction=0,
                                 rollingFriction=0, restitution=0, contactStiffness=0.0, contactDamping=0.00000,
                                 linearDamping=0.0000, angularDamping=0.00000)

        ''' Upload unripe strawberries'''

        for i in range(self.n_long_green_berries):
            berry_pose = [1.2, 1.0, 1.44]
            # Noise on position
            r = np.random.random_integers(low=0, high=3)
            if r == 0 or r == 1:
                y_noise = np.random.uniform(low=0.05, high=0.4)
            if r == 2 or r ==3:
                y_noise = np.random.uniform(low=-0.05, high=-0.4)
            berry_pose[1] += y_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or, red = False,ShortMedLong=[0,0,1])

            for i in range(0,p.getNumJoints(strawberry.strawberry)):
                p.changeDynamics(strawberry.strawberry, i,lateralFriction=0.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0,contactStiffness=0.0,contactDamping=0.00000,linearDamping =0.0000,angularDamping =0.00000)

        for i in range(self.n_short_green_berries):
            berry_pose = [1.2, 0.0, 1.44]
            # Noise on position
            r = np.random.random_integers(low=0, high=3)
            if r == 0 or r == 1:
                y_noise = np.random.uniform(low=0.05, high=0.4)
            if r == 2 or r ==3:
                y_noise = np.random.uniform(low=-0.05, high=-0.4)
            berry_pose[1] += y_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or, red = False,ShortMedLong=[1,0,0])

            for i in range(0,p.getNumJoints(strawberry.strawberry)):
                p.changeDynamics(strawberry.strawberry, i, lateralFriction=0.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0,contactStiffness=0.0,contactDamping=0.00000,linearDamping =0.0000,angularDamping =0.00000)

        for i in range(self.n_medium_green_berries):
            berry_pose = [1.2, 1.0, 1.44]
            # Noise on position
            r = np.random.random_integers(low=0, high=3)
            if r == 0 or r == 1:
                y_noise = np.random.uniform(low=0.05, high=0.4)
            if r == 2 or r ==3:
                y_noise = np.random.uniform(low=-0.05, high=-0.4)
            berry_pose[1] += y_noise
            # Noise on orientation
            berry_or = p.getQuaternionFromEuler([0, 0, np.random.uniform(low=-1.57, high=1.57)])
            strawberry = Strawberry.Strawberry(client_id=client_id, start_pose=berry_pose, start_ori=berry_or, red = False,ShortMedLong=[0,1,0])

            for i in range(0,p.getNumJoints(strawberry.strawberry)):
                p.changeDynamics(strawberry.strawberry, i, lateralFriction=0.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0,contactStiffness=0.0,contactDamping=0.00000,linearDamping =0.0000,angularDamping =0.00000)

        ''' Upload leaves '''

        for i in range(self.n_leaves_short):
                leaf_pose = [1.2, 1.0, 1.44]
                # Noise on position
                r = np.random.random_integers(low=0, high=3)
                if r == 0 or r == 1:
                    y_noise = np.random.uniform(low=0.05, high=0.4)
                if r == 2 or r == 31:
                    y_noise = np.random.uniform(low=-0.05, high=-0.4)
                leaf_pose[1] += y_noise
                # Noise on orientation
                leaf_ori = [0, 0, np.random.uniform(low=-1.57, high=1.57)]
                leaf_ori = p.getQuaternionFromEuler(leaf_ori)
                leaf = Leaf.Leaf(client_id=client_id, start_pose=leaf_pose, start_ori=leaf_ori,Short=True,Medium=False,Long=False)
                for i in range(0, p.getNumJoints(leaf.leaf)):
                    p.changeDynamics(leaf.leaf, i,lateralFriction=0.0,spinningFriction=0.0,rollingFriction=0.0,restitution=0,contactStiffness=0.0,contactDamping=0.00000,linearDamping =0.0000,angularDamping =0.00000)

        for i in range(self.n_leaves_medium):
                leaf_pose = [1.2, 1.0, 1.44]
                # Noise on position
                r = np.random.random_integers(low=0, high=3)
                if r == 0 or r == 1:
                    y_noise = np.random.uniform(low=0.05, high=0.4)
                if r == 2 or r == 31:
                    y_noise = np.random.uniform(low=-0.05, high=-0.4)
                leaf_pose[1] += y_noise
                # Noise on orientation
                leaf_ori = [0, 0, np.random.uniform(low=-1.57, high=1.57)]
                leaf_ori = p.getQuaternionFromEuler(leaf_ori)

                leaf = Leaf.Leaf(client_id=client_id, start_pose=leaf_pose, start_ori=leaf_ori,Short=False,Medium=True,Long=False)
                for i in range(0, p.getNumJoints(leaf.leaf)):
                    p.changeDynamics(leaf.leaf, i, lateralFriction=0.,spinningFriction=0.,rollingFriction=0.,restitution=0,contactStiffness=0,contactDamping=0,linearDamping =0,angularDamping =0)

        for i in range(self.n_leaves_long):
                leaf_pose = [1.2, 1.0, 1.44]
                # Noise on position
                r = np.random.random_integers(low=0, high=3)
                if r == 0 or r == 1:
                    y_noise = np.random.uniform(low=0.05, high=0.4)
                if r == 2 or r == 31:
                    y_noise = np.random.uniform(low=-0.05, high=-0.4)
                leaf_pose[1] += y_noise
                # Noise on orientation
                leaf_ori = [0, 0, np.random.uniform(low=-1.57, high=1.57)]
                leaf_ori = p.getQuaternionFromEuler(leaf_ori)
                leaf = Leaf.Leaf(client_id=client_id, start_pose=leaf_pose, start_ori=leaf_ori,Short=False,Medium=False,Long=True)
                for i in range(0, p.getNumJoints(leaf.leaf)):
                    p.changeDynamics(leaf.leaf, i, lateralFriction=0.,spinningFriction=0.,rollingFriction=0.,restitution=0,contactStiffness=0,contactDamping=0,linearDamping =0,angularDamping =0)

    def get_red_strawberry_position(self,client_id):
        n_berry = p.getNumJoints(self.red_strawberry.strawberry,physicsClientId=client_id)-1
        berry_position = p.getLinkState(self.red_strawberry.strawberry, n_berry, computeForwardKinematics=True,physicsClientId=client_id)[4]
        return np.asarray(berry_position)