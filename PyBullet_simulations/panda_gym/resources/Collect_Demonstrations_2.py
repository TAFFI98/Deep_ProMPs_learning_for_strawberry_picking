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
import Cluster
np.random.seed(22)
def Single_Cofiguration_Rec(config):
    ''''''
    '''Create GUI - connects to the physics simulation '''
    physics_client_id = p.connect(p.DIRECT)
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
                           -0.007122770030465392, 3.14, 0.8168725628418113, 0, 0]
    Cluster_position = [1.2, 0.5, 0]

    ''' Load plane,table, panda and cluster '''
    planeId = Plane.Plane(physics_client_id)
    tableId = Table.Table(table_position=[0.5, 1.0, 0.0], client_id=physics_client_id)
    ClusterId = Cluster.Cluster(Cluster_position, client_id=physics_client_id)
    ''' Step the simulation till the cluster is stable'''
    for i in range(150):
        p.stepSimulation()
    orientation_angles = np.linspace(ClusterId.app_angle_min,ClusterId.app_angle_max,num=10)
    ''' Start simulation: 10 SAMPLES FROM THE SAME CONFIGURATION '''
    for demo in range(10):
            print('Start of ',str(demo), 'demonstration for configuration ',str(config))
            ''' Load Panda arm'''
            pandaId = FrankaPanda.FrankaPanda(offset=Panda_base_position, start_joints_state=Joint_home_position,
                                              client_id=physics_client_id)
            #assign_link_colors(pandaId.panda, max_colors=3, s=0.5, v=1.)
            dump_body(pandaId.panda)
            tool_link = link_from_name(pandaId.panda, 'ee_center')
            info = PANDA_INFO
            draw_pose(Pose(), parent=pandaId.panda, parent_link=tool_link)
            joints = get_movable_joints(pandaId.panda)
            print('Joints', [get_joint_name(pandaId.panda, joint) for joint in joints])

            ''' Take image from home position with top camera '''

            top_camera_position = p.getLinkState(pandaId.panda, 19, computeForwardKinematics=True)[0]
            top_cameraEyePosition = np.asarray(top_camera_position)
            RGB, DEPTH = camera(physics_client_id, 640, 480, focal_length=1.88 * 1e-3,
                                 farVal=10, nearVal=0.2, fov=54.79, cameraEyePosition=top_cameraEyePosition, save_id=str(config)+'_'+str(demo),
                                 yaw=-90, pitch=-18.9, roll=0,
                                 renderer=p.ER_TINY_RENDERER, RGB_show=False, RGB_save=True, DEPTH_show=False,
                                 DEPTH_save=False)
            print('Images from home position collected!')

            ''' Save Depth image as numpy array '''
            np.save(currentdir +'/Data_collected/D_home_np/'+ str(config)+'_'+str(demo) +'.npy' ,DEPTH)
            # np.load(currentdir +'Data_collected/D_home_np/'+ '0_0' +'.npy' )

            ''' Take image from home position with camera 1 '''
            # camera_1_position = p.getLinkState(pandaId.panda, 15, computeForwardKinematics=True)[0]
            # cameraEyePosition_1 = np.asarray(camera_1_position)
            # RGB_1 = camera(physics_client_id, pixelWidth, pixelHeight, focal_length = 289.07*1e-3, farVal = 300, nearVal= 0.02, fov = 80, cameraEyePosition=cameraEyePosition_1,save_id='0.0', yaw= -90, pitch=0, roll=0,
            #                renderer=p.ER_TINY_RENDERER, RGB_show=False, RGB_save=False, DEPTH_show=False, DEPTH_save=False)

            ''' Take image from home position with camera 2 '''
            # camera_2_position = p.getLinkState(pandaId.panda, 16, computeForwardKinematics=True)[0]
            # cameraEyePosition_2 = np.asarray(camera_2_position)
            # RGB_2 = camera(physics_client_id, pixelWidth, pixelHeight, focal_length = 289.07*1e-3, farVal = 300, nearVal= 0.02, fov = 80, cameraEyePosition=cameraEyePosition_2,save_id='0.0', yaw = -90, pitch=0, roll=0,
            #                renderer=p.ER_TINY_RENDERER, RGB_show=False, RGB_save=False, DEPTH_show=False, DEPTH_save=False)
            ''' Compute the initial uph orientation and position'''
            (initial_uph_position , initial_uph_orientation) = get_link_pose(pandaId.panda, tool_link)
            initial_uph_orientation =np.asarray(initial_uph_orientation)

            ''' compute the desired EE position:on top of the red strawberry'''
            final_uph_position = np.asarray(ClusterId.get_red_strawberry_position(physics_client_id))
            final_uph_position[2]+=0.06
            final_uph_orientation =np.asarray(p.getEulerFromQuaternion(initial_uph_orientation))
            final_uph_orientation[2]+=orientation_angles[demo]
            final_uph_orientation =p.getQuaternionFromEuler(final_uph_orientation)
            tool_pose = (final_uph_position, final_uph_orientation)
            '''Compute the target joints values '''
            desired_joint_positions = np.zeros(shape=(9,))
            for conf in either_inverse_kinematics(pandaId.panda, info, tool_link, tool_pose, use_pybullet=False,
                                                      max_distance=INF, max_time=15, max_candidates=INF):
                    desired_joint_positions[0:7] = conf
                    desired_joint_positions[7] = -0.03
                    desired_joint_positions[8] = 0.03
            for i in range(10):
             p.stepSimulation()
            time.sleep(2)
            ''' Return to home position '''

            p.resetJointState(pandaId.panda, 0, Joint_home_position[0])
            p.resetJointState(pandaId.panda, 1, Joint_home_position[1])
            p.resetJointState(pandaId.panda, 2, Joint_home_position[2])
            p.resetJointState(pandaId.panda, 3, Joint_home_position[3])
            p.resetJointState(pandaId.panda, 4, Joint_home_position[4])
            p.resetJointState(pandaId.panda, 5, Joint_home_position[5])
            p.resetJointState(pandaId.panda, 6, Joint_home_position[6])
            p.resetJointState(pandaId.panda, 10, Joint_home_position[7])
            p.resetJointState(pandaId.panda, 11, Joint_home_position[8])
            for i in range(10):
             p.stepSimulation()
            time.sleep(2)
            (initial_uph_position , initial_uph_orientation) = get_link_pose(pandaId.panda, tool_link)
            initial_uph_position =np.asarray(initial_uph_position)
            initial_uph_orientation =np.asarray(initial_uph_orientation)
            ''' Initialize '''
            trajectory = []
            timing = []
            n_steps = 0
            ''' Difference between target and current position of the EE '''
            d1 = np.abs(np.subtract(initial_uph_position[0], final_uph_position[0]))
            d2 = np.abs(np.subtract(initial_uph_position[1], final_uph_position[1]))
            d3 = np.abs(np.subtract(initial_uph_position[2], final_uph_position[2]))

            ''' Difference between target and current orientation of the EE'''
            distance_orientation = np.min([1, np.arccos(2 * np.square(np.dot(initial_uph_orientation, final_uph_orientation)) - 1)])

            print('Start of the Simulation!!')
            ''' Step the simulation'''
            while(d2 > 0.001) or (d1 > 0.001) or (d3 > 0.051) or (distance_orientation > 0.001) :
                p.stepSimulation()
                #time.sleep(1. /100)
                n_steps += 1
                print(str(n_steps),"  Simulation Steps" )

                p.setJointMotorControlArray(pandaId.panda, [0, 1, 2, 3, 4, 5, 6, 10, 11], p.POSITION_CONTROL,
                                            targetPositions=desired_joint_positions,positionGains=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04])

                ''' Log of the 7 joints trajectories'''
                ''' TIME '''
                time_sim = n_steps * 0.01
                timing.append(time_sim)
                ''' JOINTS '''
                joint_provisional=np.zeros(shape=(9,))
                for joint in range(7):
                    joint_provisional[joint] = p.getJointState(pandaId.panda, joint, physics_client_id)[0]
                joint_provisional[7]= p.getJointState(pandaId.panda, 10, physics_client_id)[0]
                joint_provisional[8]= p.getJointState(pandaId.panda, 11, physics_client_id)[0]

                trajectory.append(joint_provisional.tolist())

                ''' Difference between target and current position of the EE '''
                (current_uph_position, current_uph_orientation) = get_link_pose(pandaId.panda, tool_link)
                d1 = np.abs(np.subtract(current_uph_position[0], final_uph_position[0]))
                d2 = np.abs(np.subtract(current_uph_position[1], final_uph_position[1]))
                d3 = np.abs(np.subtract(current_uph_position[2], final_uph_position[2]))

                ''' Difference between target and current orientation of the EE'''
                distance_orientation = np.min([1, np.arccos(2 * np.square(np.dot(current_uph_orientation, final_uph_orientation)) - 1)])
                print(d1,d2,d3,distance_orientation)
            print('End of the Simulation!!')
            ''' Reset the position of panda keeping the current cluster configuration '''
            p.removeBody(pandaId.panda)

            ''' Dump the trajectory in .json file'''

            json_file = currentdir + '/Data_collected/trajectories/'+str(config)+'/'+ str(config)+'_'+str(demo)+'.json'
            json_data = dict()
            json_data['joint_position'] = trajectory
            json_data['time'] = timing
            Path(currentdir + '/Data_collected/trajectories/'+str(config)+'/').mkdir(exist_ok=True, parents=True)
            with open(json_file, 'w') as write_file:
                  json.dump(json_data, write_file, ensure_ascii=False, indent=4)
            print('Trajectory saved as .json file')
    return physics_client_id


if __name__ == '__main__':
    ''' 10 samples for 50 different configurations are recorded '''
    print('Start of the Demonstrations Collection')
    for config in range(57,60):
        print('CONFIGURATION n:', str(config))
        physics_client_id = Single_Cofiguration_Rec(config)
        p.disconnect(physics_client_id)