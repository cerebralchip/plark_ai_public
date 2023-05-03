from gym_plark.envs.plark_env import PlarkEnv
from plark_game import classes
import numpy as np


import logging
logger = logging.getLogger(__name__)


class SuperPelicanEnv(PlarkEnv):

    def __init__(self, config_file_path=None, **kwargs):
        super(SuperPelicanEnv, self).__init__(config_file_path, image_based=False, **kwargs)
        if self.driving_agent != 'pelican':
            raise ValueError('This environment only supports pelican')
        self.image_based = False

        # get starting column and row for pelican movement reward
        self.pelican_col = self.env.activeGames[len(
            self.env.activeGames)-1].pelicanPlayer.col
        self.pelican_row = self.env.activeGames[len(
            self.env.activeGames)-1].pelicanPlayer.row

    def calculate_reward(self, action):
        reward = 0

        game = self.env.activeGames[len(self.env.activeGames)-1]
        _info = {}

        #check illegal move
        if self.driving_agent == 'pelican':
            illegal_move = game.illegal_pelican_move
        _info['illegal_move'] = illegal_move

        if illegal_move == True:
            reward = reward + self.illegal_move_reward

        # Reward for droping a sonobouy 
        if action == 'drop_buoy' and illegal_move == False:
            self.globalSonobuoys = game.globalSonobuoys
            if len(self.globalSonobuoys)>1: 
                sonobuoy = self.globalSonobuoys[-1]
                sbs_in_range = game.gameBoard.searchRadius(sonobuoy.col, sonobuoy.row, sonobuoy.range, "SONOBUOY")
                sbs_in_range.remove(sonobuoy) # remove itself from search results
                if len(sbs_in_range) > 0:
                    reward = reward + self.buoy_too_close_reward
                else:
                    reward = reward + self.buoy_far_apart_reward 
            else:
                reward = reward + self.buoy_far_apart_reward    

        # if a sonobuoy is activated, add a reward
        if game.pelicanPlayer.madmanStatus == True:
            reward = reward + 0.2

        # get to bottom area within first few turns
        # get current turn
        self.current_turn = game.turn_count

        # gets the current column and row of the pelican, after a step has been taken
        self.new_pelican_col = game.pelicanPlayer.col
        self.new_pelican_row = game.pelicanPlayer.row

        # check current turn
        # if turn is less than 3, perfrom reward check:
        if self.current_turn < 3:
            # if row hasnt changed, no reward
            # if row is lower, reward
            if self.new_pelican_row < self.pelican_row:
                reward = reward + .3
            # if row is higher, punish
            elif self.new_pelican_row > self.pelican_row:
                reward = reward - .2

        return reward, _info

    def step(self, action):

        # check if action is an integer:
        try:
            action = self.ACTION_LOOKUP[action]
        except:
        #elif action is a numpy.ndarray:
            if isinstance(action, np.ndarray):
                #get the first element of the array as int
                action_int = int(action.item())
                action = self.ACTION_LOOKUP[action_int]
            # print("Action: ", action)
            else:  # If the action is a multi-element array
                action_tuple = tuple(action.tolist())
                action = self.ACTION_LOOKUP[action_tuple]

        if self.verbose:
            logger.info('Action:'+action)
        gameState, uioutput = self.env.activeGames[len(
            self.env.activeGames)-1].game_step(action)

        self.status = gameState
        self.uioutput = uioutput

        obs = self._observation()

        reward, _info = self.calculate_reward(action)

        terminated = False
        truncated = False

        #  PELICANWIN = Pelican has won
        #  ESCAPE     = Panther has won
        #  BINGO      = Panther has won, Pelican has reached it's turn limit and run out of fuel
        #  WINCHESTER = Panther has won, All torpedoes dropped and stopped running. Panther can't be stopped
        if self.status in ["ESCAPE", "BINGO", "WINCHESTER", "PELICANWIN"]:
            if self.verbose:
                logger.info("GAME STATE IS " + self.status)
            if self.status in ["ESCAPE", "BINGO", "WINCHESTER"]:
                if self.status == "BINGO":
                    truncated = True
                else:
                    terminated = True
                if self.driving_agent == 'pelican':
                    reward = -1
                    _info['result'] = "LOSE"
                else:
                    raise ValueError('driving_agent not set correctly')
            if self.status == "PELICANWIN":
                terminated = True
                if self.driving_agent == 'pelican':
                    reward = 1
                    _info['result'] = "WIN"
                else:
                    raise ValueError('driving_agent not set correctly')

        return obs, reward, terminated, truncated, _info
