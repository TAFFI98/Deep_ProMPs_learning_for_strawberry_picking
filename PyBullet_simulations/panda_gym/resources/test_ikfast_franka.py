#!/usr/bin/env python

from __future__ import print_function

import pybullet as p

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, INF

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver,ikfast_inverse_kinematics,closest_inverse_kinematics


def test_retraction(robot, info, tool_link, distance=0.1, **kwargs):
    ik_joints = get_ik_joints(robot, info, tool_link)
    start_pose = get_link_pose(robot, tool_link)
    end_pose = multiply(start_pose, Pose(Point(z=-distance)))
    handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
    #handles.extend(draw_pose(start_pose))
    #handles.extend(draw_pose(end_pose))
    path = []
    pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
    for i, pose in enumerate(pose_path):
        print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
        handles.extend(draw_pose(pose))
        conf = next(either_inverse_kinematics(robot, info, tool_link, pose, **kwargs), None)
        if conf is None:
            print('Failure!')
            path = None
            wait_for_user()
            break
        set_joint_positions(robot, ik_joints, conf)
        path.append(conf)
        wait_for_user()
        # for conf in islice(ikfast_inverse_kinematics(robot, info, tool_link, pose, max_attempts=INF, max_distance=0.5), 1):
        #    set_joint_positions(robot, joints[:len(conf)], conf)
        #    wait_for_user()
    remove_handles(handles)
    return path

def test_ik(robot, info, tool_link, tool_pose):
    draw_pose(tool_pose)
    # TODO: sort by one joint angle
    # TODO: prune based on proximity
    ik_joints = get_ik_joints(robot, info, tool_link)
    for conf in either_inverse_kinematics(robot, info, tool_link, tool_pose, use_pybullet=False,
                                          max_distance=INF, max_time=10, max_candidates=INF):
        # TODO: profile
        set_joint_positions(robot, ik_joints, conf)
        wait_for_user()

#####################################

def main():
    connect(use_gui=True)
    add_data_path()
    draw_pose(Pose(), length=1.)
    set_camera_pose(camera_point=[1, -1, 1])

    plane = p.loadURDF("plane.urdf")

    with LockRenderer():
        with HideOutput(True):
            robot = load_pybullet(FRANKA_URDF, fixed_base=True)
            assign_link_colors(robot, max_colors=3, s=0.5, v=1.)
            #set_all_color(robot, GREEN)


    dump_body(robot)
    print('Start?')
    wait_for_user()

    info = PANDA_INFO
    tool_link = link_from_name(robot, 'ee_center')
    draw_pose(Pose(), parent=robot, parent_link=tool_link)
    joints = get_movable_joints(robot)
    print('Joints', [get_joint_name(robot, joint) for joint in joints])
    check_ik_solver(info)

    sample_fn = get_sample_fn(robot, joints)
    for i in range(10):
        print('Iteration:', i)
        conf =[0.004046583856622568, -0.21545873952713104, -0.005451456256474691, -1.9326629693958524, -0.007122770030465392,
         3.14, 0.8168725628418113, -0.03, 0.03]
        #conf = sample_fn()
        set_joint_positions(robot, joints, conf)
        wait_for_user()
        print(get_link_pose(robot, tool_link))
        test_ik(robot, info, tool_link, get_link_pose(robot, tool_link))
        #test_retraction(robot, info, tool_link, use_pybullet=False,
                        #max_distance=0.1, max_time=0.05, max_candidates=100)
    disconnect()

if __name__ == '__main__':
    main()
