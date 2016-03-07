'''
Contains different agents that inherit from the random agent.

State space
(pos_x, pos_y, breeze, smell, num_flash) == agent.state_

Note that actions start at 1, resulting in some awkward code.
'''

import numpy as np
from wumpus_text import Agent, Action

# Convert moving direction to flashlight direction
move_to_flash = {
    Action.UP: Action.FLASH_UP,
    Action.DOWN: Action.FLASH_DOWN,
    Action.LEFT: Action.FLASH_LEFT,
    Action.RIGHT: Action.FLASH_RIGHT}

moves = {
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT}

flashes = {
    Action.FLASH_UP,
    Action.FLASH_DOWN,
    Action.FLASH_LEFT,
    Action.FLASH_RIGHT}


class FeedbackAgent(Agent):
    def nextState(self, s, reward):
        Agent.nextState(self, s, reward)
        #print 'Setting next state {}, reward {}'.format(s, reward)


class EngineeredAgent(Agent):
    '''
    Almost like random agent, but do not use
    flashlight if no smell
    '''
    def reset(self):
        self.last_dir = np.random.choice(list(moves))

    def getAction(self):
        x, y, breeze, smell, num_flash = self.state_
        if smell and num_flash:  # likely that wumpus is ahead of us, use flashlight if possible
            action = Action(np.random.choice(list(moves.union(flashes))))
        else:  # just move, useless to use flashlight
            action = Action(np.random.choice(list(moves)))
        # Update last taken direction
        if action in moves:
            self.last_dir = action
        return action

def ravel(coords, dims):
    '''
    Flatten indices using C-style order
    '''
    id = coords[0]
    for s, dim in zip(coords[1:], dims[:-1]):
        id += s
    return id


class StateEncoding(object):
    '''
    Abstract encoding of state.

    Any state is a tuple that can be unraveled
    to a single integer ID using state_dims
    '''
    def __init__(self, state_dims):
        self.state_dims = state_dims

    def get_state(self, agent):
        '''return observed state'''
        pass

    def get_state_id(self, agent):
        state = self.get_state(agent)
        state_id = ravel(state, self.state_dims)
        return state_id


class BSF(StateEncoding):
    def __init__(self, n_flash):
        StateEncoding.__init__(self, [2, 2, n_flash])

    def get_state(self, agent):
        return agent.state_[2:]


class ABSF(StateEncoding):
    def __init__(self, n_flash):
        StateEncoding.__init__(self, [len(Action), 2, 2, n_flash])

    def get_state(self, agent):
        return [agent.last_action-1] + agent.state_[2:]


class XYBSF(StateEncoding):
    def __init__(self, grid_size, n_flash):
        StateEncoding.__init__(self, [grid_size, grid_size, 2, 2, n_flash])

    def get_state(self, agent):
        return agent.state_


class EpsilonGreedy(Agent):
    '''
    With probability epsilon explore random action,
    o/w take greedy action.

    epsilon==1:    Completely random
    epsilon==0:    Completely greedy
    '''
    def __init__(self, epsilon, encoding):
        self.epsilon = epsilon
        self.encoding = encoding
        self.cum_rewards = np.zeros((np.prod(encoding.state_dims), len(Action)))  # q[s, a]
        self.n_visits = 1. * np.ones_like(self.cum_rewards)  # number of visits, 1 by default
        Agent.__init__(self)

    def reset(self):
        self.first_visit = True # First visit of episode

    def getAction(self):
        if np.random.uniform() < self.epsilon:
            # random action
            action = Agent.getAction(self)
        else:
            # greedy action
            state_id = self.encoding.get_state_id(self)
            q_s = self.cum_rewards[state_id] / self.n_visits[state_id]
            action = Action(1 + np.argmax(q_s))
        self.last_action = action
        return action

    def nextState(self, s, reward):
        # update Q
        if not self.first_visit:
            state_id = self.encoding.get_state_id(self)
            self.cum_rewards[state_id, self.last_action-1] += reward
            self.n_visits[state_id, self.last_action-1] += 1.
        else:
            self.first_visit = False
        # update internal state
        Agent.nextState(self, s, reward)



