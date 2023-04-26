import os, subprocess, time, signal
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import numpy as np 

import sys
from plark_game import classes

from gym_plark.envs.plark_env import PlarkEnv

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)

class PSuperPlark(PlarkEnv):

	def __init__(self,config_file_path=None,verbose=False,  **kwargs):
		super(SuperPlark, self).__init__(config_file_path,verbose,**kwargs)
		if self.driving_agent != 'pelican':
			raise ValueError('This environment only supports pelican')
		
		self.pelican_col = self.env.activeGames[len(self.env.activeGames)-1].pelicanPlayer.col
		self.pelican_row = self.env.activeGames[len(self.env.activeGames)-1].pelicanPlayer.row
  
	def step(self, action):
		
		action = self.ACTION_LOOKUP[action]
		if self.verbose:
			logger.info('Action:'+action)
		gameState,uioutput = self.env.activeGames[len(self.env.activeGames)-1].game_step(action)

		self.status = gameState
		self.uioutput = uioutput 
     

		game = self.env.activeGames[len(self.env.activeGames) -1]
		#check for illegal move
		if self.driving_agent == 'pelican':
			illegal_move = game.illegal_pelican_move
		# punish bad move
		if illegal_move == True: 
			reward = -1
						
        #set my rewards here
        

		ob = self._observation()
		
		terminated = False
		truncated = False

		if self.status in ["PELICANWIN","ESCAPE","BINGO","WINCHESTER"]:
			if self.status == "BINGO": #pelican ran out of fuel so timeout
				truncated = True
			else:
				terminated = True
			if self.verbose:
				logger.info("GAME STATE IS " + self.status)   
		
		return ob, reward, terminated, truncated, []
