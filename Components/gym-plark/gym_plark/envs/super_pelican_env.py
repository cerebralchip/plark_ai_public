from gym_plark.envs.plark_env import PlarkEnv
from plark_game import classes



import logging
logger = logging.getLogger(__name__)


class SuperPelicanEnv(PlarkEnv):

    def __init__(self, config_file_path=None, **kwargs):
        super(SuperPelicanEnv, self).__init__(config_file_path, **kwargs)
        if self.driving_agent != 'pelican':
            raise ValueError('This environment only supports pelican')
        self.madman_reward = 0.5

    def step(self, action):
        action = self.ACTION_LOOKUP[action]
        if self.verbose:
            logger.info('Action:'+action)
        gameState, uioutput = self.env.activeGames[len(
            self.env.activeGames)-1].game_step(action)
        self.status = gameState
        self.uioutput = uioutput

        obs = self._observation()

        reward = 0
        _info = {}

        game = self.env.activeGames[len(self.env.activeGames)-1]

        if self.driving_agent == 'pelican':
            illegal_move = game.illegal_pelican_move
        _info['illegal_move'] = illegal_move
        if illegal_move == True:
            reward = reward + self.illegal_move_reward

        #if a sonobuoy is red, add a reward
        if self.game.pelicanPlayer.madmanStatus == True:
            reward = reward + self.madman_reward

        if self.driving_agent == 'pelican':  # If it wasn't an illegal move.
            # Reward for droping a sonobouy
            if action == 'drop_buoy' and illegal_move == False:
                self.globalSonobuoys = game.globalSonobuoys
                if len(self.globalSonobuoys) > 1:
                    sonobuoy = self.globalSonobuoys[-1]
                    sbs_in_range = game.gameBoard.searchRadius(
                        sonobuoy.col, sonobuoy.row, sonobuoy.range, "SONOBUOY")
                    # remove itself from search results
                    sbs_in_range.remove(sonobuoy)
                    if len(sbs_in_range) > 0:
                        reward = reward + self.buoy_too_close_reward
                    else:
                        reward = reward + self.buoy_far_apart_reward
                else:
                    reward = reward + self.buoy_far_apart_reward

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
