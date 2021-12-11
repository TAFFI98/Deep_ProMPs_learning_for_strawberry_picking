import cv2
import pathlib
import numpy as np
from typing import List
from natsort import natsorted
import os
import tensorflow_probability as tfp
import json
import numpy.matlib as mat
import matplotlib.pyplot as plt
import math
import scipy
import tensorflow as tf
from scipy.stats import norm
from ProMP import ProMP,ProMPTuner
from pathlib import Path
from statsmodels.stats.correlation_tools import cov_nearest
import rosbag
tfd = tfp.distributions
np.set_printoptions(threshold=np.inf)
from plotting_functions import plot_traj_distribution_1_joint,plot_weights_distributions_1_joint
import config as cfg
def bag_to_json(datset_dir_for_single_config,json_path):
    '''
    Function to convert .bag files to .json files
    :param datset_dir_for_single_config: directory of the .bag files
    :param json_path: directory in which the .json images are going to be stored
    '''
    # Create output directory
    Path(json_path).mkdir(exist_ok=True, parents=True)

    for idx, bag_file in enumerate(pathlib.Path(datset_dir_for_single_config).rglob('*.bag')):

        bag = rosbag.Bag(bag_file.as_posix())
        json_file = json_path+'experiment_' + str(idx) + ".json"
        json_data = dict()
        json_data['joint_position'] = []
        json_data['joint_speed'] = []
        json_data['joint_torque'] = []
        json_data['time'] = []

        for topic, msg, t in bag.read_messages(topics=["/joint_states"]):
            json_data['time'].append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            json_data['joint_position'].append(msg.position)
            json_data['joint_speed'].append(msg.velocity)
            json_data['joint_torque'].append(msg.effort)
        # subtract first time sample for all
        json_data['time'] = [sample - json_data['time'][0] for sample in json_data['time']]

        with open(json_file , 'w') as write_file:
          json.dump(json_data, write_file, ensure_ascii=False, indent=4)
        print('Saved as JSON file')

def bgr_to_rgb(img_dir,save_dir_img):
    '''
    Function to convert BGR images to RGB images
    :param img_dir:  directory in which the BGR images are going to be stored
    :param save_dir_img:  directory in which the RGB images will be stored
    '''
    for idx, image_file in enumerate(Path(img_dir).rglob('*.png')):
     print(image_file.as_posix())
     bgr = cv2.imread(image_file.as_posix())
     rgb= cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
     #Visualization of the output image.
     #cv2.imshow('original_file', rgb)
     #cv2.waitKey(0)
     cv2.imwrite(save_dir_img+str(idx+1)+'.png',rgb)

def is_pos_semidef(x,tol=0):
    return np.all(np.linalg.eigvals(x) >= tol)

def is_pos_def(x,tol=0):
    return np.all(np.linalg.eigvals(x) > tol)

def check_symmetric(a, tol=0):
    return np.all(np.abs(a-a.T) < tol)

def PRoMPs_extraction(N_BASIS, N_DOF, N_T, json_files_path, save_file_path):
    n_t = N_T
    n_joints = N_DOF
    promp = ProMP(N_BASIS, N_DOF, N_T)
    config = int(Path(json_files_path).parts[-1])
    json_files = [pos_json for pos_json in os.listdir(json_files_path) if pos_json.endswith('.json')]
    json_files = natsorted(json_files)
    all_weights = []
    for index, js in enumerate(json_files):
        with open(os.path.join(json_files_path, js)) as json_file:
            json_text = json.load(json_file)
            if 'joint_position' in json_text.keys():
                joint_pos_key = 'joint_position'
            elif 'joint_pos' in json_text.keys():
                joint_pos_key = 'joint_pos'
            else:
                raise KeyError(f"Joint position not found in the trajectory file {js}")
            joint_pos = json_text[joint_pos_key]
            joint_matrix = np.vstack(np.array(joint_pos)[:, 6])
            num_samples = joint_matrix.shape[0]
            phi = promp.basis_func_gauss_local(num_samples)  # (150,8)
            weights = np.transpose(mat.matmul(np.linalg.pinv(phi), joint_matrix))
            all_weights.append(np.hstack((weights)))
    all_weights = np.asarray(all_weights)  # (10 ,56)
    num_samples = all_weights.shape[0]
    t = np.empty(shape=(n_t, n_joints, num_samples), dtype='float64')
    for i in range(num_samples):
        t[:, :, i] = promp.trajectory_from_weights(all_weights[i, :], vector_output=False)  # (150, 7, 10)
    MEAN_TRAJ = np.mean(t, axis=-1, dtype='float64')  # (150,1)
    t = np.empty(shape=(n_t * n_joints, num_samples), dtype='float64')  # (1050)
    for i in range(num_samples):
        t[:, i] = promp.trajectory_from_weights(all_weights[i, :], vector_output=True)  # (1050,34)
    COV_TRAJ = np.cov(t, dtype='float64')
    # COV_TRAJ = COV_TRAJ + 1e-15*np.identity(1050)
    STD_TRAJ = promp.get_std_from_covariance(COV_TRAJ)
    STD_TRAJ = np.reshape(STD_TRAJ, (n_t, -1), order='F')  # (1050)
    # print('The trajectory covariance matrix is positive definite?   ', is_pos_def(COV_TRAJ))

    # ProMPs
    MEAN_WEIGHTS = promp.get_mean_from_weights(all_weights).astype('float64')  # (56,)
    all_weights = np.transpose(all_weights)  # (56,34)
    COV_WEIGHTS = promp.get_cov_from_weights(all_weights)  # (56,56)
    # COV_WEIGHTS = COV_WEIGHTS+ 1e-10*np.identity(56)
    COV_WEIGHTS = COV_WEIGHTS.astype('float64')
    STD_WEIGHTS = promp.get_std_from_covariance(COV_WEIGHTS)
    print('The weights covariance matrix is positive definite?   ', is_pos_def(COV_WEIGHTS))
    MEAN_TRAJ_PROMP = promp.trajectory_from_weights(MEAN_WEIGHTS, vector_output=False)  # (150, 7)
    COV_TRAJ_PROMP = promp.get_traj_cov(COV_WEIGHTS).astype('float64')  # (150, 150)
    # COV_TRAJ_PROMP = COV_TRAJ_PROMP + 1e-11 * np.identity(1050)
    COV_TRAJ_PROMP = COV_TRAJ_PROMP.astype('float64')
    STD_TRAJ_PROMP = promp.get_std_from_covariance(COV_TRAJ_PROMP)  # (1050)
    STD_TRAJ_PROMP = np.reshape(STD_TRAJ_PROMP, (n_t, -1), order='F')
    # print('The ProMPs traj covariance matrix is positive definite?   ', is_pos_def(COV_TRAJ_PROMP))

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config, mean_traj_1=MEAN_TRAJ_PROMP,
                                   traj_std_1=STD_TRAJ_PROMP, mean_traj_2=MEAN_TRAJ, traj_std_2=STD_TRAJ, show=False,
                                   save=True)
    # plot_weights_distributions_1_joint(save_file_path=save_file_path,mean_weights_1=MEAN_WEIGHTS,std_weights_1=STD_WEIGHTS,n_func=8,mean_weights_2=np.zeros(shape=(8,)), std_weights_2 = np.zeros(shape=(8,)),show = False,save = False)
    return MEAN_TRAJ_PROMP, COV_TRAJ_PROMP, MEAN_WEIGHTS, COV_WEIGHTS


if __name__ == '__main__':
    '''
    1. Transform .bag to.json files
    '''
    # for i in range(51):
    #  datset_dir_for_single_config='/Users/alessandratafuro/Documents/Lincoln projects/M. Sc. Thesis/code/data/bag_files/' +str(i) +'/'
    #  json_path='/Users/alessandratafuro/Documents/Lincoln projects/M. Sc. Thesis/code/data/json_files/' +str(i) +'/'
    #  bag_to_json(datset_dir_for_single_config, json_path)

    '''
    2. Transform .bag to.json files
    '''
    N_BASIS = 8
    N_DOF = 1
    N_T = 150
    promp = ProMP(N_BASIS, N_DOF, N_T)
    for i in range(60):
        json_file_path = cfg.ROOT_DIR + '/data/json_files/' + str(i) + '/'
        save_file_path = cfg.EXP_DIR + '/PLOTS/traj & weights true distributions/'
        MEAN_TRAJ, COV_TRAJ, MEAN_WEIGHTS, COV_WEIGHTS = PRoMPs_extraction(N_BASIS, N_DOF, N_T,
                                                                           json_files_path=json_file_path,
                                                                           save_file_path=save_file_path)

        # Build a lower triangular matrix, with the diagonal values of log(D) and the lower values of L
        L, D, _ = scipy.linalg.ldl(COV_WEIGHTS)
        d = np.diag(D)
        L_new = L
        L_new[np.diag_indices(8)] = d
        tril_elements = tfp.math.fill_triangular_inverse(L_new)

        # SAVE ANNOTATION
        config = int(Path(json_file_path).parts[-1])
        annotation = {}
        save_annotation_path = cfg.ANNOTATION_PATH
        annotation["mean_weights"] = np.vstack(MEAN_WEIGHTS).tolist()
        annotation["L"] = tril_elements.numpy().tolist()
        annotation["configuration"] = config
        dump_file_name = str(config) + '.json'
        dump_file_path = save_annotation_path + dump_file_name
        with open(dump_file_path, 'w') as f:
            json.dump(annotation, f)
