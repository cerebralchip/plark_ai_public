import os
import subprocess
import time
import signal
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np

import sys
from plark_game import classes

from gym_plark.envs.plark_env import PlarkEnv

import logging
logger = logging.getLogger(__name__)


class PlarkEnvSparse(PlarkEnv):

    def __init__(self,config_file_path=None, **kwargs):
        super(PlarkEnvSparse, self).__init__(config_file_path, **kwargs)
        
  
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

        ob = self._observation()

        reward = 0
        terminated = False
        truncated = False
        _info = {}

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
                elif self.driving_agent == 'panther':
                    reward = 1
                    _info['result'] = "WIN"
                else:
                    raise ValueError('driving_agent not set correctly')
            if self.status == "PELICANWIN":
                terminated = True
                if self.driving_agent == 'pelican':
                    reward = 1
                    _info['result'] = "WIN"
                elif self.driving_agent == 'panther':
                    reward = -1
                    _info['result'] = "LOSE"
                else:
                    raise ValueError('driving_agent not set correctly')

        return ob, reward, terminated, truncated, _info

