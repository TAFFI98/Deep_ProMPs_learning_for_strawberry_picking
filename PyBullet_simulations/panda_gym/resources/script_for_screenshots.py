import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
from pathlib import Path
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver,ikfast_inverse_kinematics,closest_inverse_kinematics

import pybullet as p
import json
import numpy as np
from camera import camera
import time
import FrankaPanda
import Table
import Plane
import Cluster_l as Cluster
np.random.seed(24)

def main():
    ''''''
    '''Create GUI - connects to the physics simulation '''
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.resetDebugVisualizerCamera(1.08, 132.04, -20.22, [0.7, 0.69, 0.89])
    draw_pose(Pose(), length=1.)

    ''' Resets the simulation '''
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 1000, physics_client_id)

    ''' Define cartesian position for the base of the Panda arm, the joints Home values and the Cluster'''
    Panda_base_position = [0.3, 1, 0.625]
    Joint_home_position = [0.004046583856622568, -0.21545873952713104, -0.005451456256474691, -1.9326629693958524,
                           -0.007122770030465392, 3.14, 0.8168725628418113,-0.03,0.03]
    Cluster_position = [1.2, 0.5, 0]

    ''' Load plane,table, panda and cluster '''
    planeId = Plane.Plane(physics_client_id)
    tableId = Table.Table(table_position=[0.5, 1.0, 0.0], client_id=physics_client_id)
    ClusterId = Cluster.Cluster(Cluster_position, client_id=physics_client_id)
    ''' Step the simulation till the cluster is stable'''
    for i in range(50):
        p.stepSimulation()
    pandaId = FrankaPanda.FrankaPanda(offset=Panda_base_position, start_joints_state=Joint_home_position,
                                          client_id=physics_client_id)
    #assign_link_colors(pandaId.panda, max_colors=3, s=0.5, v=1.)
    dump_body(pandaId.panda)
    tool_link = link_from_name(pandaId.panda, 'ee_center')

    ''' camera '''
    top_camera_position = p.getLinkState(pandaId.panda, 19, computeForwardKinematics=True)[0]
    top_cameraEyePosition = np.asarray(top_camera_position)
    RGB, DEPTH = camera(physics_client_id, 640, 480, focal_length=1.88 * 1e-3,
                        farVal=10, nearVal=0.2, fov=54.79, cameraEyePosition=top_cameraEyePosition,
                        save_id= '_' ,
                        yaw=-90, pitch=-18.9, roll=0,
                        renderer=p.ER_TINY_RENDERER, RGB_show=False, RGB_save=False, DEPTH_show=False,
                        DEPTH_save=False)


    ''' compute the desired EE position: in front of the red strawberry'''
    berry_position = np.asarray(ClusterId.get_red_strawberry_position(physics_client_id))
    berry_position[2]+=0.09
    (uph_position , uph_orientation) = get_link_pose(pandaId.panda, tool_link)
    uph_orientation =np.asarray(p.getEulerFromQuaternion(uph_orientation))
    uph_orientation[2]+=ClusterId.app_angle_min
    uph_orientation =p.getQuaternionFromEuler(uph_orientation)
    '''Move to the red berry '''
    info = PANDA_INFO
    draw_pose(Pose(), parent=pandaId.panda, parent_link=tool_link)
    joints = get_movable_joints(pandaId.panda)
    print('Joints', [get_joint_name(pandaId.panda, joint) for joint in joints])
    check_ik_solver(info)
    ik_joints = get_ik_joints(pandaId.panda, info, tool_link)

    tool_pose = (berry_position, uph_orientation)
    joint_positions = np.zeros(shape=(9,))
    while p.isConnected():
        # for conf in either_inverse_kinematics(pandaId.panda, info, tool_link, tool_pose, use_pybullet=False,
        #                                       max_distance=INF, max_time=10, max_candidates=INF):
        #     joint_positions[0:7] = conf
        #     joint_positions[7] = -0.03
        #     joint_positions[8] = 0.03
        #
        #
        # print('--------')
        p.stepSimulation()




if __name__ == '__main__':
    main()