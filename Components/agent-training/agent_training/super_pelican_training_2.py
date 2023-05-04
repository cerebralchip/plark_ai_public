# imports
from gym_plark.envs import plark_env, super_pelican_env
from stable_baselines3 import PPO, DQN
from sb3_contrib import TRPO, MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def trainer():

    path = "/home/alexr/dev/plark/plark_ai_public"

    # make env
    # env = super_pelican_env.SuperPelicanEnv(
    #     driving_agent="pelican",
    #     config_file_path=path
    #     + "/Components/plark-game/plark_game/game_config/30x30/balanced.json",
    # )

    env = SubprocVecEnv(
        [
            lambda: super_pelican_env.SuperPelicanEnv(
                driving_agent="pelican",
                config_file_path=path + "/Components/plark-game/plark_game/game_config/30x30/balanced.json",
                random_panther_start_position=True,
                max_illegal_moves_per_turn=10,
            )
            for _ in range(4)
        ]
    )
    # make model

    model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO('CnnPolicy', env, verbose=1)

    # model = DQN('MlpPolicy', env, verbose=1)

    # model = TRPO("MlpPolicy", env, verbose=1)
    # model = TRPO.load(path+"/data/agents/models/TRPO_super_pelican",env, verbose=1)


    # model = MaskablePPO('MlpPolicy', env, verbose=1)

    modelType = "PPO"

    # model learn
    model.learn(1000000, log_interval=1, progress_bar=True)

    # save model to /home/alexr/dev/plark/plark_ai_public/data/agents/models/agent_name
    savepath = path + "/data/agents/models/" + modelType + "_super_pelican"
    model.save(savepath)

if __name__ == '__main__':
    trainer()
