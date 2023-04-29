from .agent import Agent

class Panther_Agent(Agent):
    def action_lookup(self, action):
        #Usually action is an int, but sometimes it is a numpy array (I have found when doing selfplay.py)
        try:
            return ACTION_LOOKUP[action]
        except:
            action_int = action.tolist()
            return ACTION_LOOKUP[action_int]
    
            

ACTION_LOOKUP = {
    0 : '1',  # Up
    1 : '2',  # Up right
    2 : '3',  # Down right
    3 : '4',  # Down
    4 : '5',  # Down left
    5 : '6',  # Up left
    6 : 'end'
}
