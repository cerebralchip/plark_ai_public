# imports
from gym_plark.envs import plark_env, super_pelican_env
from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import monitor
import sys
import pathlib

def trainer(modelType):

    #set a variable to be the path to the current dir
    path = str(pathlib.Path().resolve())
    savepath = path + "/data/agents/models/" + modelType + "_super_pelican"


    # make env
    env = SubprocVecEnv(
        [
            lambda: 
            monitor.Monitor(
                super_pelican_env.SuperPelicanEnv(
                    driving_agent="pelican",
                    config_file_path=path + "/Components/plark-game/plark_game/game_config/20x20/balanced.json",
                    random_panther_start_position=True,
                    max_illegal_moves_per_turn=10,
                ),
                filename=savepath,
                allow_early_resets=True,
            )            
            for _ in range(16)
        ]
    )

    # make model
    # type 'tensorboard --logdir $savepath to see tensorboard

    if modelType == "PPO":
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=savepath)
    elif modelType == "DQN":
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=savepath)
    elif modelType == "A2C":
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=savepath)
    elif modelType == "TRPO":
        model = TRPO('MlpPolicy', env, verbose=1, tensorboard_log=savepath)

    # model learn
    model.learn(1000000, progress_bar=True)


    # save model
    model.save(savepath)

if __name__ == '__main__':
    modelType = str(sys.argv[1])
    trainer(modelType)
