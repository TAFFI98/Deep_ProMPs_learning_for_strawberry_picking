import gym
import time
from ProMP.utils import plot_sampled_trajectories
import numpy as np
from ProMP.ProMP_framework import ProMP
import panda_gym
import cv2
np.random.seed(6)
def main():
    env = gym.make('panda-gym-v0',GUI_on=True)
    for i in range(0,1):

        ''' Reset the environment '''
        initial_observation, RGB_home, DEPTH_home= env.reset()
        print( "----- INITIAL OBSERVATION -----")
        print(" Strawberry position: ", initial_observation[-3:])
        print(" EE position: ", initial_observation[0:3])
        print(" EE orientation: ", initial_observation[3:6])

        ''' Show RGB image'''
        #cv2.imshow('RGB image', cv2.cvtColor(RGB_home, cv2.COLOR_BGR2RGB))
        #cv2.waitKey()

        ''' ProMP intialization'''
        N_BASIS = 8
        N_DOF = 1
        N_JOINTS = 9
        N_T = 150
        promp = ProMP(N_BASIS, N_DOF, N_T)

        ''' Predict mean and covariance of ProMP weights '''
        weights_mean_pred, weights_cov_pred = env.predict_weights_mean_and_covariance(RGB_home,DEPTH_home)

        ''' Sample weights'''
        theta_sampled = weights_mean_pred
        # theta_sampled = np.empty(shape=weights_mean_pred.shape)
        #
        # for j in range(N_JOINTS):
        #      theta_sampled[j, :] = np.random.multivariate_normal(weights_mean_pred[j, :], weights_cov_pred[j, :,:], 1)


        theta_sampled[0, :] = np.asarray(               [4.83674393e+00 ,- 2.24142947e+01 , 5.22900709e+01, - 8.11657618e+01,
         9.36090292e+01 ,- 7.99593890e+01 , 4.77494150e+01 ,- 1.53189421e+01])
        theta_sampled[1, :] = np.asarray(   [-2.35886985e+00 ,- 6.79855964e-01 , 8.53750528e+00, - 1.32144069e+01,
         1.29099368e+01 ,- 7.29762087e+00 , 2.63801715e+00 , 1.28046535e-01])
        theta_sampled[2, :] = np.asarray(         [-6.97130878e+00 , 3.21532802e+01 ,- 7.27024753e+01 , 1.08186457e+02,
         - 1.18018096e+02 , 9.55160512e+01, - 5.41360553e+01 , 1.67186614e+01])
        theta_sampled[3, :] = np.asarray(          [-1.11337006e+00, - 1.60251418e+01  ,3.86234097e+01 ,- 6.34936922e+01,
         6.52086915e+01 ,- 5.20047905e+01,  2.41508835e+01 ,- 8.16126263e+00])
        theta_sampled[4, :] = np.asarray(           [7.05845298e+00 ,- 3.26963952e+01  ,8.11223325e+01, - 1.36300159e+02,
         1.72026399e+02 ,- 1.60731928e+02 , 1.05644589e+02, - 3.80780415e+01])
        theta_sampled[5, :] = np.asarray(                   [-5.40075319e+00 , 3.42271839e+01, - 5.94884759e+01  ,9.28825216e+01,
         - 9.39248313e+01 , 8.16188749e+01 ,- 4.11549248e+01 , 1.66160748e+01])
        theta_sampled[6, :] = np.asarray(            [8.55527617e+00 ,- 2.04528881e+01 , 3.51556506e+01 ,- 4.14147545e+01,
         3.69468773e+01 ,- 1.76654268e+01  ,1.63456533e+00 , 3.94937245e+00])

        ''' Reconstruct trajectory '''
        sampled_trajectories = np.zeros(shape=(N_JOINTS,N_T))
        for j in range(N_JOINTS):
            sampled_trajectories[j,:] = np.squeeze(promp.trajectory_from_weights(theta_sampled[j,:], vector_output=False))

        ''' Plot the reconstructed trajectory'''
        plot_sampled_trajectories(save_path='/Users/alessandratafuro/Desktop/', sampled_traj=sampled_trajectories, show=False, save=False)

        ''' Compute the reward '''
        policy_final_reward = 0
        for i in range(N_T):
            target_pos = sampled_trajectories[:,i]
            observation, reward, done, info = env.step(target_pos)
            print('reward: ', reward)
            print(info)
            policy_final_reward += reward
            time.sleep(env.timeStep)
        print('Reward of mean policy:   ', policy_final_reward)

    ''' Close the environment'''
    env.close()

if __name__ == '__main__':
    main()
