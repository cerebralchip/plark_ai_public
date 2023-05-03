from stable_baselines3.common.env_checker import check_env
import gym_plark.envs as envs

env = envs.PlarkEnv(driving_agent='pelican', config_file_path='/home/alexr/dev/plark/plark_ai_public/Components/plark-game/plark_game/game_config/30x30/panther_easy.json')
# It will check your custom environment and output additional warnings if needed
check_env(env)
print('done')