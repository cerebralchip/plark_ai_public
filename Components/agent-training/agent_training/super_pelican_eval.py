#imports
from gym_plark.envs import plark_env, plark_env_sparse, super_pelican_env
from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO, MaskablePPO
import helper

modelType = "PPO"
path = "/home/alexr/dev/plark/plark_ai_public/data/agents/models/"+modelType+"_super_pelican"

#load agent
model = PPO.load(path)
# model = DQN.load(path)
# model = TRPO.load(path)
# model = MaskablePPO.load(path)

#evaluate agent
env = plark_env_sparse.PlarkEnvSparse(driving_agent='pelican', config_file_path='/home/alexr/dev/plark_ai_public/data/configs/balanced.json')
#make video
basewidth, hsize = helper.make_video(model,env,path,n_steps = 10000,fps=10,deterministic=False,basewidth = 512,verbose =False)
