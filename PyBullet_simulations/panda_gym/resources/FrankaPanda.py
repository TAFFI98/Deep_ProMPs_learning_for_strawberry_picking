import os,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ProMPdir = os.path.dirname(currentdir)+'/ProMP/'
os.sys.path.insert(0, ProMPdir)
import pybullet as p
import numpy as np
os.sys.path.insert(0, currentdir)
from suppress_warning import suppress_stdout
from  pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF

class FrankaPanda:
	def __init__(self, offset, start_joints_state,client_id):
		self.franka_urdf_folder = currentdir + "/urdf/panda/panda_UPH.urdf"
		with suppress_stdout():
			self.panda = p.loadURDF(self.franka_urdf_folder , offset, [0,0,0,1], useFixedBase=True,physicsClientId=client_id)

		''' Set start joints positions  '''
		p.resetJointState(self.panda, 0, start_joints_state[0])
		p.resetJointState(self.panda, 1, start_joints_state[1])
		p.resetJointState(self.panda, 2, start_joints_state[2])
		p.resetJointState(self.panda, 3, start_joints_state[3])
		p.resetJointState(self.panda, 4, start_joints_state[4])
		p.resetJointState(self.panda, 5, start_joints_state[5])
		p.resetJointState(self.panda, 6, start_joints_state[6])
		p.resetJointState(self.panda, 10, start_joints_state[7])
		p.resetJointState(self.panda, 11, start_joints_state[8])
		p.changeVisualShape(self.panda,9,rgbaColor=[0,0,0,0])

		tool_link = link_from_name(self.panda, 'ee_center')
		self.initial_EE_position,self.initial_EE_orientation = get_link_pose(self.panda, tool_link)

	def apply_action_with_fingers(self,action,client_id):
		'''Applies a certain joints position'''
		p.setJointMotorControlArray(self.panda, [0,1,2,3,4,5,6,10,11], p.POSITION_CONTROL,targetPositions = action,physicsClientId=client_id)


	def apply_action_without_fingers(self,action,client_id):
		'''Applies a certain joints position'''
		p.setJointMotorControlArray(self.panda, [0,1,2,3,4,5,6], p.POSITION_CONTROL,targetPositions = action,physicsClientId=client_id)




















