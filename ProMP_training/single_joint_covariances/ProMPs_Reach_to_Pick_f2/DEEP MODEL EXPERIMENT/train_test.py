import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
import math
import cv2
from plotting_functions import plot_traj_distribution_1_joint,plot_weights_distributions_1_joint
from scipy.stats import norm
from datasets import Dataset_RGBProMP
from experiments import Experiment
from losses import KL,KL_traj,RMSE
from models import deep_ProMPs_2dmodel_RGBD
from output import plot_loss,My_metric
from ProMP import ProMP
import sys
import json
import config as cfg
import tensorflow_probability as tfp
tfd = tfp.distributions
from LDL_decomposition import LD_reconstruct,LD_from_elements_pred,LD_from_elements_true
from data_preprocessing import is_pos_def

class Experiment_ProMPs(Experiment):
    def __init__(self):
        super().__init__(cfg)

        ''' ProMPs Variables '''
        self.N_BASIS = 8
        self.N_DOF = 1
        self.N_T = 150
        self.promp = ProMP(self.N_BASIS, self.N_DOF, self.N_T)


        # Load the dataset.
        print("Loading data...")
        self.dataset = Dataset_RGBProMP(dataset_dir=cfg.ANNOTATION_PATH,rgb_dir=cfg.IMAGE_PATH,depth_dir=cfg.DEPTH_PATH)
        self.dataset.prepare_data(self.cfg.val_frac, self.cfg.N_test, self.cfg.random_state, self.cfg.use_val_in_train)
        print("Done!")

        # Load the model
        #self.model = deep_ProMPs_2dmodel()
        self.model = deep_ProMPs_2dmodel_RGBD()

        #CHOOSE OPTIMIZER
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)

        # Select loss function
        self.loss=RMSE(self.promp)

        # Callbacks.
        early_stopping = EarlyStopping(monitor="val_loss", min_delta=cfg.es["delta"], patience=cfg.es["patience"], verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=cfg.rl["factor"],patience=cfg.rl["patience"], min_lr=cfg.rl["min_lr"])
        self.callbacks = []
        if cfg.es["enable"]:
            self.callbacks.append(early_stopping)
        if cfg.rl["enable"]:
            self.callbacks.append(reduce_lr)


    def train(self):
        if self.callbacks is None:
            self.callbacks = []

        # Load the data.
        (X_train, y_train), (X_val, y_val), (_, _) = self.dataset.data
        RGB_train = X_train["RGB"]
        D_train = X_train["D"]


        RGB_val = X_val["RGB"]
        D_val = X_val["D"]
        mean_train = np.asarray(y_train['mean_weights'])
        mean_val = y_val['mean_weights']
        yt =mean_train
        yv =mean_val
        print('RGB train:   ', np.shape(RGB_train))
        print('D train:   ', np.shape(D_train))
        print('mean train:   ', mean_train.shape)
        print('yt train:   ', yt.shape)

        print('RGB val:   ', np.shape(RGB_val))
        print('D val:   ', np.shape(D_val))
        print('mean val:   ', mean_val.shape)
        print('yt val:   ', yv.shape)

        # load the models
        #self.model.build(input_shape=(self.cfg.batch, 480, 640, 3))
        self.model.build(input_shape=[(self.cfg.batch, 256, 256, 3),(self.cfg.batch, 256, 256, 3)])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        # Train.
        history = self.model.fit([RGB_train,D_train], yt, epochs=self.cfg.epochs, batch_size=self.cfg.batch,validation_data=([RGB_val,D_val], yv), callbacks=self.callbacks)
        #Save the model.
        Path(self.cfg.MODEL_PATH).mkdir(exist_ok=True, parents=True)
        Path(self.cfg.LOSS_PATH).mkdir(exist_ok=True, parents=True)
        self.model.save(self.cfg.MODEL_PATH)
        # Save plot of training loss.
        plot_loss(self.cfg.LOSS_PATH, history.history['loss'], name='loss.png',val_loss=history.history['val_loss'])

    def eval(self, load_model_name):

        loss=self.loss
        model_load_path = os.path.join(self.cfg.MODEL_FOLDER, load_model_name)
        model= tf.keras.models.load_model(model_load_path, custom_objects={loss.__name__:loss})
        (_, _), (_, _), (X_true, y_true) = self.dataset.data
        RGB_true = X_true["RGB"]
        D_true = X_true["D"]
        ypred = model.predict([RGB_true,D_true])

        print('Tested configurations:',self.dataset.data_names['test_ids'])

        mean_true = np.asarray(y_true['mean_weights']).astype('float64')
        mean_pred = ypred          # (n_batch, 56)


        n_test=mean_true.shape[0]
        metric=0.0
        for i in range(n_test):

            MEAN_TRAJ_true = self.promp.trajectory_from_weights(mean_true[i,:], vector_output=False)
            MEAN_TRAJ_pred = self.promp.trajectory_from_weights(mean_pred[i,:], vector_output=False)


            # PLOT THE OUTPUT
            plot_traj_distribution_1_joint(save_path=os.path.join(self.cfg.OUTPUT_PATH,load_model_name), config=self.dataset.data_names['test_ids'][i], mean_traj_1=MEAN_TRAJ_true,traj_std_1=np.zeros(shape=(150,)),mean_traj_2=MEAN_TRAJ_pred,show = False, save =True)
            #plot_weights_distributions_1_joint(save_file_path=self.cfg.OUTPUT_PATH,mean_weights_1=mean_true[i,:],std_weights_1=STD_WEIGHTS_true,n_func=8,mean_weights_2=mean_pred[i,:],std_weights_2=STD_WEIGHTS_pred,show=True,save=False)

            # COMPUTE THE METRIC
            metric+=My_metric(MEAN_TRAJ_pred, MEAN_TRAJ_true, np.zeros(shape=(150,150)),np.zeros(shape=(150,150)))

        metric=metric/n_test
        # SAVE THE METRIC
        annotation = {}
        annotation["RMSE"] = str(metric)
        dump_file_path =os.path.join(self.cfg.METRIC_PATH,load_model_name) + '/metric.json'
        print('The average RMSE is:  ',annotation["RMSE"])
        Path(os.path.join(self.cfg.METRIC_PATH,load_model_name)).mkdir(exist_ok=True, parents=True)
        with open(dump_file_path, 'w') as f:
            json.dump(annotation, f)


if __name__ == "__main__":
    '''
    DEFINE THE EXPERIMENT
    '''
    Experiment_ProMPs = Experiment_ProMPs()
    '''
    TRAIN THE MODEL
    '''
    #Experiment_ProMPs.train()
    '''
    TEST THE MODEL
    '''
    Experiment_ProMPs.eval(load_model_name='model_09_12__19_29')


