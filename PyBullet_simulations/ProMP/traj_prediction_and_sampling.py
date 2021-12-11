import os,inspect
from ProMP.ProMP_framework import ProMP
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, currentdir)
import config as cfg
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
from utils import RMSE,LD_reconstruct,LD_from_elements_pred,plot_traj_distribution_1_joint,is_pos_def
import cv2
from skimage.transform import resize
class ProMPs_prediction():
    def __init__(self):
        self.cfg = cfg
        ''' ProMPs Variables. '''
        self.N_BASIS = 8
        self.N_DOF = 1
        self.N_JOINTS = 9
        self.N_T = 150
        self.promp = ProMP(self.N_BASIS, self.N_DOF, self.N_T)
        ''' Loss used to train deep models '''
        self.loss = RMSE(self.promp)

    def predict_and_sample_traj(self,RGB,DEPTH):
            loss = self.loss
            ''' Directories of the 7 different models '''
            directories_models = [self.cfg.MODEL_FOLDER_0, self.cfg.MODEL_FOLDER_1, self.cfg.MODEL_FOLDER_2,
                                  self.cfg.MODEL_FOLDER_3, self.cfg.MODEL_FOLDER_4, self.cfg.MODEL_FOLDER_5,
                                  self.cfg.MODEL_FOLDER_6,self.cfg.MODEL_FOLDER_f1,self.cfg.MODEL_FOLDER_f2]

            ''' Predict the panda joints trajectories  '''
            trajectories=[]
            for i in range(7):
                print('Predicting joint ' + str(i))

                ''' Load the model '''
                model_load_path = directories_models[i]
                model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})

                ''' Predict mean and covariance of weights of ProMPs '''
                ypred = model.predict([RGB,DEPTH])
                weights_mean_pred = ypred[..., 0:-36]  # (n_batch, 56)
                lower_tri_elements_pred = ypred[..., -36:]  # (n_batch, 1596)
                L_pred, D_pred = LD_from_elements_pred(lower_tri_elements_pred)
                weights_cov_pred = LD_reconstruct(L_pred, D_pred)
                print('The predicted weights covariance matrix is positive definite?   ', is_pos_def(weights_cov_pred))

                ''' Sample some weights from the distribution '''
                sample_of_weights = np.random.multivariate_normal( np.squeeze(weights_mean_pred), np.squeeze(weights_cov_pred), 1)
                ''' Reconstruct the correspondent trajectory '''
                sample_of_traj = np.squeeze(self.promp.trajectory_from_weights(sample_of_weights, vector_output=False)) # (1, 150)

                trajectories.append(sample_of_traj)     #(7, 150)

            ''' Predict the 2 fingers trajectories  '''
            for i in range(7,9):
                print('Predicting joint ' + str(i))

                ''' Load the model '''
                model_load_path = directories_models[i]
                model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})
                ''' Predict mean and covariance of weights of ProMPs '''
                weights_mean_pred = model.predict([RGB,DEPTH])
                ''' Sample some weights from the distribution '''
                sample_of_weights = np.random.multivariate_normal(np.squeeze(weights_mean_pred),np.zeros(shape=(8,8)), 1)
                ''' Reconstruct the correspondent trajectory '''
                sample_of_traj = np.squeeze(self.promp.trajectory_from_weights(sample_of_weights, vector_output=False))  # (1, 150)

                trajectories.append(sample_of_traj)  # (7, 150)


            return np.asarray(trajectories)

    def predict_weights_mean_and_covariance(self,RGB,D):
            ''''''
            ''' Images pre processing '''
            DEPTH = np.zeros(shape=(480, 640, 3))
            for i in range(3):
                DEPTH[:, :, i] = D
            DEPTH = np.expand_dims(resize(DEPTH,(256, 256, 3)), axis=0).astype('float64')
            RGB = np.expand_dims(resize(RGB,(256, 256, 3)), axis=0).astype('float64')

            loss = self.loss
            ''' Directories of the 7 different models '''
            directories_models = [self.cfg.MODEL_FOLDER_0, self.cfg.MODEL_FOLDER_1, self.cfg.MODEL_FOLDER_2,
                                  self.cfg.MODEL_FOLDER_3, self.cfg.MODEL_FOLDER_4, self.cfg.MODEL_FOLDER_5,
                                  self.cfg.MODEL_FOLDER_6,self.cfg.MODEL_FOLDER_f1,self.cfg.MODEL_FOLDER_f2]

            ''' Predict the panda joints trajectories  '''
            weights_mean_pred_all_joints = np.zeros(shape=(self.N_JOINTS, self.N_BASIS))
            weights_cov_pred_all_joints = np.zeros(shape=(self.N_JOINTS, self.N_BASIS, self.N_BASIS))
            for i in range(7):
                print('Predicting joint ' + str(i))
                ''' Load the model '''
                model_load_path = directories_models[i]
                model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})
                ''' Predict mean and covariance of weights of ProMPs '''
                ypred = model.predict([RGB,DEPTH])
                weights_mean_pred = ypred[..., 0:-36]
                lower_tri_elements_pred = ypred[..., -36:]
                L_pred, D_pred = LD_from_elements_pred(lower_tri_elements_pred)
                weights_cov_pred = LD_reconstruct(L_pred, D_pred)
                weights_mean_pred_all_joints[i,:] = np.squeeze(weights_mean_pred)
                weights_cov_pred_all_joints[i,:,:]= np.squeeze(weights_cov_pred)

            ''' Predict the 2 fingers trajectories  '''
            for i in range(7,9):
                print('Predicting joint ' + str(i))
                ''' Load the model '''
                model_load_path = directories_models[i]
                model = tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__: loss})
                ''' Predict mean and covariance of weights of ProMPs '''
                weights_mean_pred = model.predict([RGB, DEPTH])
                weights_mean_pred_all_joints[i, :] = np.squeeze(weights_mean_pred)
                weights_cov_pred_all_joints[i,:,:]= np.squeeze(np.zeros(shape=(8,8)))

            return weights_mean_pred_all_joints,weights_cov_pred_all_joints
