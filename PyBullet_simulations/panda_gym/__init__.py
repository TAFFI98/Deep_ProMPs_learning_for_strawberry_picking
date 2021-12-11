from gym.envs.registration import register

# Adds the environment to the Gyms registry
# Allows for the creation of the environment through the standard method of gym.make()

register(
    id='panda-gym-v0', 
    entry_point='panda_gym.envs:PandaGymEnv'
)

