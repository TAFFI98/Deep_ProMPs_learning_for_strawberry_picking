import gym
import time
from PoWER.PoWER import PoWER_class
import panda_gym
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
np.random.seed(6)
def main():
    ''' ProMP weights optimization by POWER algorithm '''
    '''Reset the target environment '''
    env_target = gym.make('panda-gym-v0',GUI_on = False)
    initial_observation, RGB_home, DEPTH_home = env_target.reset()
    '''Predict the weights distribution from the RGB image '''
    weights_mean_pred, weights_cov_pred = env_target.predict_weights_mean_and_covariance(RGB_home,DEPTH_home)

    weights_mean_pred_7j = weights_mean_pred[0:7,:]
    weights_cov_pred_7j = weights_cov_pred[0:7,:,:]
    ''' Visualize RGB image '''
    #cv2.imshow('RGB image', cv2.cvtColor(RGB_home, cv2.COLOR_BGR2RGB))
    #cv2.waitKey()


    ''' Initialize PoWER algorithm '''
    PoWER = PoWER_class()
    theta_optimal = weights_mean_pred_7j
    covariance_optimal = weights_cov_pred_7j

    success_history =[]
    reward_history = []
    theta_optimal_history = []
    theta_optimal_history.append(weights_mean_pred)
    for i in range(1500):
         print('Iteration of target policy n: ', i+1)
         theta_optimal, covariance_optimal, optimal_policy_reward,success = PoWER.PoWER_algorithm(predicted_mean_w = theta_optimal, predicted_cov_w = covariance_optimal,env=env_target)
         reward_history.append(optimal_policy_reward[-1])
         success_history.append(success)
         distance = np.zeros(7)
         for j in range(7):
             distance[j] = euclidean(theta_optimal_history[i][j, :], theta_optimal[j, :])
         '''Check convergence '''
         if np.mean(distance) < 0.00001:
            print('Convergence reached')
            break
         ''' Check success '''
         if success:
            print('Success reached')
            theta_optimal = theta_optimal_history[i]
            break
         else:
          theta_optimal_history.append(theta_optimal)
          print(theta_optimal)
    ''' Plot the Reward of target policy '''
    print(reward_history)
    print('------')
    print(theta_optimal)
    plt.plot(reward_history)
    plt.show()

if __name__ == '__main__':
    main()