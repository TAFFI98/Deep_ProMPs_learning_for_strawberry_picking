import gym
import numpy as np
import pybullet as p
from scipy.spatial.distance import euclidean
from skimage.transform import resize

from ProMP import traj_prediction_and_sampling
from ProMP.utils import plot_sampled_trajectories

from panda_gym.resources.camera import camera
from panda_gym.resources.FrankaPanda import FrankaPanda
from panda_gym.resources.Cluster import Cluster
from panda_gym.resources.Cluster_l import Cluster as Cluster_l
from panda_gym.resources.Cluster_r import Cluster as Cluster_r
from panda_gym.resources.Plane import Plane
from panda_gym.resources.Table import Table


from  panda_gym.resources.pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF


class PandaGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, GUI_on = True):

        self.timeStep = 1 / 1000
        if GUI_on:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(self.timeStep,self.client)

        ''' Seed parameter '''
        self.np_random, _ = gym.utils.seeding.np_random()

        ''' 
        Action space - 7 dimensional: 
        -9 joints position values 
        '''
        self.action_space =gym.spaces.box.Box(
            low = np.array([-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-0.03,-0.03]),
            high = np.array([2*np.pi,2*np.pi,2*np.pi,2*np.pi,2*np.pi,2*np.pi,2*np.pi,0.03,0.03]))

        ''' 
        Observation space - 10 dimensional:
        - position of the end-effector (X,Y,Z) +
        - orientation of the EE in quaternions (q1,q2,q3,q4) +
        - position of the target strawberry (X,Y,Z)
        '''
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-3,-3,-3,-1,-1,-1,-1,-3,-3,-3]),
            high = np.array([3, 3, 3, 1, 1, 1,1, 3, 3, 3]))

        ''' Panda agent + Cluster, Table, Plane variables '''
        self.Panda = None
        self.Cluster = None
        self.Table = None
        self.Plane = None

        '''RGB image dimensions Realsense430 '''
        self.pixelWidth_top_camera = 640
        self.pixelHeight_top_camera = 480

        ''' Gym variables '''
        self.done = False
        self.reset()
        self.table_collision = False
        # self.stem_collision = False

        ''' Desired EE position variables '''
        self.strawberry_position = None
        self.desired_ee_position = None
        self.f_x = None
        self.f_y = None
        self.f_z = None
        self.f_xy = None
        self.initial_distance = None
        self.current_distance = None

    def step(self, action):

        self.Panda.apply_action_without_fingers(action,self.client)

        ''' Step the simulation by a single step '''
        p.stepSimulation(self.client)

        ''' Get the EE position '''
        tool_link = link_from_name(self.Panda.panda, 'ee_center')
        EE_position, EE_orientation = get_link_pose(self.Panda.panda, tool_link)


        ''' Define observations from simulation '''
        observation = np.concatenate(( EE_position,EE_orientation,self.strawberry_position), axis = None)

        ''' Computation of the distance of the EE from the desired EE position '''

        ee_berry_distance_x = np.abs(np.subtract(self.desired_ee_position[0], EE_position[0]))
        ee_berry_distance_y = np.abs(np.subtract(self.desired_ee_position[1], EE_position[1]))
        ee_berry_distance_z = np.abs(np.subtract(self.desired_ee_position[2], EE_position[2]))

        ee_berry_distance_xy = np.sqrt(np.sum(np.square(ee_berry_distance_x) + np.square(ee_berry_distance_y)))
        ee_berry_distance_tot = np.sqrt(np.sum(np.square(ee_berry_distance_xy) + np.square(ee_berry_distance_z)))

        EE_inclination = np.asarray(p.getEulerFromQuaternion(EE_orientation))
        EE_inclination[2] = 0.1
        EE_inclination_desired = np.asarray(p.getEulerFromQuaternion(self.Panda.initial_EE_orientation))
        EE_inclination_desired[2] = 0.1

        EE_inclination_desired =p.getQuaternionFromEuler(EE_inclination_desired)
        EE_inclination = p.getQuaternionFromEuler(EE_inclination)

        distance_orientation = np.min([1, np.arccos(2 * np.square(np.dot(EE_inclination, EE_inclination_desired)) - 1)])

        ''' Check if the gripper is colliding with the stem '''
        p.performCollisionDetection()
        collision_table = p.getContactPoints(self.Panda.panda, self.Table.table, physicsClientId = self.client)
        collision_stem = p.getContactPoints(self.Panda.panda, self.Cluster.red_strawberry.strawberry, 10,11 ,physicsClientId=self.client)


        ''' REWARD COMPUTATION '''
        success = False

        ''' Case in which the gripper is colliding with the table '''
        ''' Case in which the gripper is colliding with the table '''
        if self.table_collision == True:
            reward = 0
        elif self.table_collision == False and bool(collision_table) == True:
            self.table_collision = True
            reward = 0
        elif bool(collision_table) == False and ee_berry_distance_xy <= 0.005 and ee_berry_distance_z <= 0.005 and ee_berry_distance_tot == self.current_distance and distance_orientation <= 0.01:
            success = True
            factor_dist = 8
            reward = factor_dist

        elif bool(collision_table) == False and ee_berry_distance_xy <= 0.005 and ee_berry_distance_z <= 0.005 and ee_berry_distance_tot == self.current_distance and distance_orientation > 0.01:
            factor_dist = 8
            reward = factor_dist - np.exp(distance_orientation)

        elif bool(collision_table) == False and ee_berry_distance_xy <= 0.005 and ee_berry_distance_z <= 0.005 and ee_berry_distance_tot < self.current_distance:
            factor_dist = 8
            reward = factor_dist - np.exp(ee_berry_distance_tot - self.current_distance) - np.exp(distance_orientation)

        elif bool(collision_table) == False and ee_berry_distance_xy <= 0.005 and ee_berry_distance_z > 0.005 and ee_berry_distance_tot <= self.current_distance:
            factor_dist = 8
            reward = factor_dist - np.exp(ee_berry_distance_z) - np.exp(distance_orientation)
            if bool(collision_stem) == True:
                reward= 0.1*reward

        elif bool(collision_table) == False and ee_berry_distance_xy > 0.005 and ee_berry_distance_z <= 0.005 and ee_berry_distance_tot <= self.current_distance:
            factor_dist = 8
            reward = factor_dist - np.exp(ee_berry_distance_xy) - np.exp(distance_orientation)
            if bool(collision_stem) == True:
                reward= 0.1*reward

        elif bool(collision_table) == False and ee_berry_distance_xy > 0.005 and ee_berry_distance_z > 0.005 and ee_berry_distance_tot <= self.current_distance:
            # else the reward id inversely proportional to the euclidean distance of the EE from the strawberry
            factor_dist = 8
            reward = factor_dist - np.exp(ee_berry_distance_xy) - np.exp(ee_berry_distance_z) - np.exp(distance_orientation)
            if bool(collision_stem) == True:
                reward= 0.1*reward

        elif ee_berry_distance_tot > self.current_distance:
            reward = 0

        info = {'Table collision': self.table_collision,'EE-berry distance': ee_berry_distance_tot, 'Success':success}
        self.current_distance = ee_berry_distance_tot
        return np.array(observation, dtype=np.float32), float(reward), self.done, info

    def seed(self, seed=None):
        # Random seeding for generating random positions for the strawberry
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        ''''''
        ''' Seed to fix the cluster configuration'''
        p.resetSimulation(self.client)
        ''' Resets the simulation '''
        p.setTimeStep(1 / 1000, self.client)
        p.resetDebugVisualizerCamera(3.1, 90, -30, [0.0, 1, 0.0],physicsClientId=self.client)
        p.setGravity(0, 0, -9.81,physicsClientId=self.client)

        ''' Define cartesian position for the base of the Panda arm, the joints Home values and the Cluster'''
        Panda_base_position = [0.3, 1, 0.625]
        Joint_home_position = [0.004046583856622568, -0.21545873952713104, -0.005451456256474691, -1.9326629693958524, -0.007122770030465392, 3.14, 0.8168725628418113, 0, 0]
        Cluster_position = [1.2, 0.5, 0]
        '''Load plane,table, panda and cluster'''
        self.Plane = Plane(client_id=self.client)
        self.Table = Table(table_position = [0.5, 1.0, 0.0],client_id=self.client)
        self.Panda = FrankaPanda( offset = Panda_base_position, start_joints_state = Joint_home_position,client_id=self.client)
        r = np.random.random_integers(low=0, high=3)
        if r == 0 :
            self.Cluster = Cluster(Cluster_position, client_id=self.client)
        if r == 1:
            self.Cluster = Cluster_r(Cluster_position, client_id=self.client)
        if r == 2:
            self.Cluster = Cluster_l(Cluster_position, client_id=self.client)

        ''' Get the EE and Strawberry positions '''
        tool_link = link_from_name(self.Panda.panda, 'ee_center')
        EE_position, EE_orientation = get_link_pose(self.Panda.panda, tool_link)
        self.strawberry_position = self.Cluster.get_red_strawberry_position(self.client)
        self.desired_ee_position = self.strawberry_position
        self.desired_ee_position[2] += 0.06

        self.f_x = abs(self.desired_ee_position[0] - self.Panda.initial_EE_position[0])
        self.f_y = abs(self.desired_ee_position[1] - self.Panda.initial_EE_position[1])
        self.f_z = abs(self.desired_ee_position[2] - self.Panda.initial_EE_position[2])
        self.f_xy = np.sqrt(np.sum(np.square(self.f_x) + np.square(self.f_y)))
        self.initial_distance = np.sqrt(np.sum(np.square(self.f_xy) + np.square(self.f_z)))
        self.current_distance = self.initial_distance

        ''' Observations from sim '''
        observation = np.concatenate(( EE_position, EE_orientation, self.strawberry_position), axis = None)

        '''Take image from home position '''
        top_camera_position = p.getLinkState(self.Panda.panda, 19, computeForwardKinematics=True,physicsClientId=self.client)[0]
        top_cameraEyePosition = np.asarray(top_camera_position)
        RGB, DEPTH = camera(self.client, self.pixelWidth_top_camera, self.pixelHeight_top_camera, focal_length=1.88 * 1e-3,
                            farVal=10, nearVal=0.2, fov=54.79, cameraEyePosition=top_cameraEyePosition,
                            save_id='',
                            yaw=-90, pitch=-18.9, roll=0,
                            renderer=p.ER_TINY_RENDERER, RGB_show=False, RGB_save=False, DEPTH_show=False,
                            DEPTH_save=False)

        return np.array(observation, dtype=np.float32), RGB, DEPTH

    def reset_panda(self):
        p.removeBody(self.Panda.panda,physicsClientId=self.client)

        Panda_base_position = [0.3, 1, 0.625]
        Joint_home_position = [0.004046583856622568, -0.21545873952713104, -0.005451456256474691, -1.9326629693958524,
                               -0.007122770030465392, 3.14, 0.8168725628418113, 0, 0]

        self.Panda = FrankaPanda(offset=Panda_base_position, start_joints_state=Joint_home_position, client_id=self.client)

        ''' Get the EE and Strawberry positions '''
        tool_link = link_from_name(self.Panda.panda, 'ee_center')
        EE_position, EE_orientation = get_link_pose(self.Panda.panda, tool_link)
        self.strawberry_position = self.Cluster.get_red_strawberry_position(self.client)
        self.desired_ee_position = self.strawberry_position
        self.desired_ee_position[2] += 0.06

        self.f_x = abs(self.desired_ee_position[0] - self.Panda.initial_EE_position[0])
        self.f_y = abs(self.desired_ee_position[1] - self.Panda.initial_EE_position[1])
        self.f_z = abs(self.desired_ee_position[2] - self.Panda.initial_EE_position[2])
        self.f_xy = np.sqrt(np.sum(np.square(self.f_x) + np.square(self.f_y)))
        self.initial_distance = np.sqrt(np.sum(np.square(self.f_xy) + np.square(self.f_z)))
        self.current_distance = self.initial_distance

        ''' Observations from sim '''
        observation = np.concatenate((EE_position, EE_orientation, self.strawberry_position), axis=None)

        return np.array(observation, dtype=np.float32)

    def predict_trajectory(self,RGB_home,DEPTH_home):
        ''''''
        '''
        Given the RGB image of the target from the home position predict a trajectory distribution, 
        sample and perform the sampled one
        '''
        ProMPs_prediction = traj_prediction_and_sampling.ProMPs_prediction()
        sampled_trajectories = ProMPs_prediction.predict_and_sample_traj(RGB_home,DEPTH_home)

        ''' Plot the sampled trajectories '''
        #plot_sampled_trajectories(save_path='', sampled_traj=sampled_trajectories, show=True, save=False)

        return sampled_trajectories

    def predict_weights_mean_and_covariance(self,RGB_home,DEPTH_home):
        ''''''
        '''
        Given the RGB image of the target from the home position predict weights mean and covariance
        '''
        ProMPs_prediction = traj_prediction_and_sampling.ProMPs_prediction()
        weights_mean_pred, weights_cov_pred = ProMPs_prediction.predict_weights_mean_and_covariance(RGB_home,DEPTH_home)
        return weights_mean_pred, weights_cov_pred

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

