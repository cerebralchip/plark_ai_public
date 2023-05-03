#imports
from gym_plark.envs import plark_env, super_pelican_env
from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO, MaskablePPO

#make env
env = super_pelican_env.SuperPelicanEnv(driving_agent='pelican', config_file_path='/home/alexr/dev/plark/plark_ai_public/Components/plark-game/plark_game/game_config/30x30/balanced.json')
#make model
model = PPO('CnnPolicy', env, verbose=1)
# model = DQN('MlpPolicy', env, verbose=1)
# model = TRPO('MlpPolicy', env, verbose=1)
# model = MaskablePPO('MlpPolicy', env, verbose=1)

modelType = "PPO"

#model learn
model.learn(10000)

#save model to /home/alexr/dev/plark/plark_ai_public/data/agents/models/agent_name
path = "/home/alexr/dev/plark/plark_ai_public/data/agents/models/"+modelType+"_super_pelican"
model.save(path)