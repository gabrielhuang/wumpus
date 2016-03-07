'''
Contains different agents that inherit from the random agent.

State space
(pos_x, pos_y, breeze, smell, num_flash) == agent.state_

Note that actions start at 1, resulting in some awkward code.
'''

import numpy as np
import scipy.stats
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
    for s, dim in zip(coords[1:], dims[1:]):
        id = id * dim + s
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
        StateEncoding.__init__(self, [2, 2, n_flash+1])

    def get_state(self, agent):
        return agent.state_[2:]


class ABSF(StateEncoding):
    def __init__(self, n_flash):
        StateEncoding.__init__(self, [len(Action), 2, 2, n_flash+1])

    def get_state(self, agent):
        return [agent.last_action-1] + agent.state_[2:]
        #return [0] + agent.state_[2:]


class XYBSF(StateEncoding):
    def __init__(self, grid_size, n_flash):
        StateEncoding.__init__(self, [grid_size, grid_size, 2, 2, n_flash+1])

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
        # do not actually reset training stuff
        self.first_visit = True # First visit of episode
        self.current_action = Agent.getAction(self)  # pick random fake previous action

    def getAction(self):
        '''
        !!!DO NOT OVERLOAD THIS FUNCTION!!!

        Return action and memorize in self.current_action
        '''
        self.current_action = self.getActionReal()
        return self.current_action

    def getActionReal(self):
        if np.random.uniform() < self.epsilon:
            # random action
            action = Agent.getAction(self)
        else:
            # greedy action
            state_id = self.encoding.get_state_id(self)
            q_s = self.cum_rewards[state_id] / self.n_visits[state_id]
            action = Action(1 + np.argmax(q_s))
        # save current action for use in nextState
        return action

    def nextState(self, s, reward):
        # update Q
        if not self.first_visit:
            state_id = self.encoding.get_state_id(self)
            self.cum_rewards[state_id, self.current_action-1] += reward
            self.n_visits[state_id, self.current_action-1] += 1.
        else:
            self.first_visit = False
        # update internal state
        Agent.nextState(self, s, reward)
        self.last_action = self.current_action  # current action is now last action



def softmax(u):
    '''
    Return exp(u) / exp(u).flatten()
    '''
    v = np.exp(u - u.max())
    return v / v.sum()


class Softmax(EpsilonGreedy):
    def __init__(self, temperature, encoding):
        Agent.__init__(self)
        EpsilonGreedy.reset(self)
        self.temperature = temperature
        self.encoding = encoding
        self.cum_rewards = np.zeros((np.prod(encoding.state_dims), len(Action)))  # q[s, a]
        self.n_visits = 1. * np.ones_like(self.cum_rewards)  # number of visits, 1 by default

    def getActionReal(self):
        # sample
        state_id = self.encoding.get_state_id(self)
        q_s = self.cum_rewards[state_id] / self.n_visits[state_id]
        dist = softmax(q_s / self.temperature)
        rvs = scipy.stats.rv_discrete(name='cust', values=(range(len(Action)), dist))
        action = Action(1 + rvs.rvs())
        return action


class UCB(EpsilonGreedy):
    def __init__(self, lbda, encoding):
        Agent.__init__(self)
        EpsilonGreedy.reset(self)
        self.lbda = lbda
        self.encoding = encoding
        self.cum_rewards = np.zeros((np.prod(encoding.state_dims), len(Action)))  # q[s, a]
        self.n_visits = 0 * np.ones_like(self.cum_rewards)  # number of visits, 1 by default

    def getActionReal(self):
        # sample
        state_id = self.encoding.get_state_id(self)
        q_s = self.cum_rewards[state_id] / self.n_visits[state_id]
        scores = q_s + self.lbda * np.sqrt(2 *
                    np.log(1 + self.n_visits[state_id].sum())/(1 + self.n_visits[state_id]))
        action = Action(1 + np.argmax(scores))
        return action

    def nextState(self, s, reward):
        scaled_reward = (float(reward) + 1) / 101  # reward must be between 0 and 1 for UCB
        EpsilonGreedy.nextState(self, s, scaled_reward)

