#imports
from gym_plark.envs import plark_env, plark_env_sparse, super_pelican_env
from stable_baselines3 import PPO, DQN
# from sb3_contrib import TRPO, MaskablePPO
import helper

modelType = "PPO"
path = "/home/alexr/dev/plark/plark_ai_public"



#load agent
model = PPO.load(path+"/data/agents/models/"+modelType+"_super_pelican")
# model = DQN.load(path)
# model = TRPO.load(path)
# model = MaskablePPO.load(path)

#evaluate agent
env = plark_env_sparse.PlarkEnvSparse(driving_agent='pelican', config_file_path=path+'/Components/plark-game/plark_game/game_config/30x30/balanced.json', image_based=False)

#evaluate agent
mean_reward, std_reward, victories = helper.evaluate_policy(model, env, n_eval_episodes=10)
print("mean_reward: "+str(mean_reward)+"\nstd_reward: "+str(std_reward)+"\nvictories: "+str(victories)+"\n")


#make video
vidpath = path+"/data/agents/models/"+modelType+"_super_pelican.mp4" 
basewidth, hsize = helper.make_video(model,env,vidpath,n_steps = 8000,fps=10,deterministic=False,basewidth = 512,verbose =False)
