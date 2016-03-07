#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple program for studying RL algorithm on the Wumpus world 
CeCILL License

created by Gaetan Marceau Caron [01/02/2016]
            
Usage: wumpus [-i <flag>] [-t <flag>] [-w <flag>] [-v <flag>] [-g <size>] [-n <int>] [-e <int>]

Options:
-h --help      Show the description of the program
-i <flag> --hmi <flag>  a flag for activating the graphical interface [default: True]
-t <flag> --tore <flag>  a flag for choosing the tore grid [default: True]
-w <flag> --wumpus_dyn <flag>  a flag for activating thw Wumpus moves (beware!) [default: False]
-v <flag> --verbose <flag>  a flag for activating the verbose mode [default: True]
-g <size> --grid_size <size>  an integer for the grid size [default: 4] 
-n <int> --n_flash <int>  an integer for the number of power units [default: 5] 
-e <int> --max_n_episode <int>  the maximum number of episode [default: 100]
"""

from tkinter import * 
from PIL import Image, ImageTk
import numpy as np
from time import sleep
from enum import IntEnum, unique

from docopt import docopt

@unique
class Action(IntEnum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    FLASH_UP = 5
    FLASH_DOWN = 6
    FLASH_LEFT = 7
    FLASH_RIGHT = 8

class Agent:
    def __init__(self):
        self.reset()
        
    def reset(self):
        pass
        
    def getAction(self):
        return Action(np.random.randint(1,len(Action)))
    
    def getPosition(self):
        return self.state_[:2]

    def getState(self):
        return self.state_

    def nextState(self,s,reward):
        self.state_ = s

class Environment:

    def __init__(self, agent, my_args=None):
        self.grid_size_ = (int(my_args["--grid_size"]),int(my_args["--grid_size"]))
        self.hole_pos_ = (1,1)
        self.treasure_pos_ = (self.grid_size_[0]-1,self.grid_size_[1]-1)

        self.agent = agent

        self.DEFAULT_N_FLASH = int(my_args["--n_flash"])
        self.DEFAULT_REWARD = -1.
        self.KILL_REWARD = 5.
        self.TREASURE_REWARD = 100.
        self.HOLE_REWARD = -10.
        self.WUMPUS_REWARD = -10.
        self.TORE_TOPO = (my_args["--tore"]=="True")
        self.DYN_WUMPUS = (my_args["--wumpus_dyn"]=="True")

        self.reset()

    def reset(self):
        self.agent.reset()
        init_state = self.getInitState()
        self.agent.nextState(init_state, 0.)
        self.wumpus_pos_ = [1,2]

    def getInitState(self):
        return [0,0,0,0,self.DEFAULT_N_FLASH]
        
    def getGridSize(self):
        return self.grid_size_

    def getWumpusPosition(self):
        return self.wumpus_pos_

    def getHolePosition(self):
        return self.hole_pos_

    def getTreasurePosition(self):
        return self.treasure_pos_

    def getNFlash(self):
        return self.n_flash_
    
    def moveWumpus(self):
        a = Action(np.random.randint(1,4)) #Warning hardcoded value!
        self.wumpus_pos_ = self.moveAgent(self.wumpus_pos_,a)

    def moveAgent(self, curr_pos, a):
        next_pos = []
        if a == Action.UP:
            next_pos = [curr_pos[0], curr_pos[1]+1]
        elif a == Action.DOWN:
            next_pos = [curr_pos[0], curr_pos[1]-1]
        elif a == Action.LEFT:
            next_pos = [curr_pos[0]-1, curr_pos[1]]
        elif a == Action.RIGHT:
            next_pos = [curr_pos[0]+1, curr_pos[1]]

        if not self.TORE_TOPO:
            next_pos = [min(self.grid_size_[0]-1,next_pos[0]),  min(self.grid_size_[1]-1,next_pos[1])]
            next_pos = [max(0,next_pos[0]),  max(0,next_pos[1])]
        else:
            if next_pos[0] == self.grid_size_[0]:
                next_pos = [0, next_pos[1]]
            elif next_pos[0] == -1:
                next_pos = [self.grid_size_[0]-1, next_pos[1]]

            if next_pos[1] == self.grid_size_[1]:
                next_pos = [next_pos[0], 0]
            elif next_pos[1] == -1:
                next_pos = [next_pos[0], self.grid_size_[1]-1]

        return next_pos 
            
    def flashAgent(self, s, a):
        agent_pos = s[:2]
        n_flash = s[4]
        if n_flash > 0 and self.wumpus_pos_[0] >= 0:
            if a == Action.FLASH_UP:
                if self.wumpus_pos_[0] == agent_pos[0] and self.wumpus_pos_[1] == agent_pos[1]+1:
                    self.wumpus_pos_ = [-1,-1]
                    return True
            elif a == Action.FLASH_DOWN:
                if self.wumpus_pos_[0] == agent_pos[0] and self.wumpus_pos_[1] == agent_pos[1]-1:
                    self.wumpus_pos_ = [-1,-1]
                    return True
            elif a == Action.FLASH_LEFT: 
                if self.wumpus_pos_[0] == agent_pos[0]-1 and self.wumpus_pos_[1] == agent_pos[1]:
                    self.wumpus_pos_ = [-1,-1]
                    return True
            elif a == Action.FLASH_RIGHT:
                if self.wumpus_pos_[0] == agent_pos[0]+1 and self.wumpus_pos_[1] == agent_pos[1]:
                    self.wumpus_pos_ = [-1,-1]
                    return True
        return False

    def testForEnd(self, s):
        agent_pos = s[:2]
        if self.wumpus_pos_[0] == agent_pos[0] and self.wumpus_pos_[1] == agent_pos[1]:
            return (self.WUMPUS_REWARD, True)
        elif self.hole_pos_[0] == agent_pos[0] and self.hole_pos_[1] == agent_pos[1]:
            return (self.HOLE_REWARD, True)
        elif self.treasure_pos_[0] == agent_pos[0] and self.treasure_pos_[1] == agent_pos[1]:
            return (self.TREASURE_REWARD, True)
        else:
            return (0, False)
        
    def updateSense(self, agent_pos):
        [smell,breeze] = [0,0]
        if abs(self.wumpus_pos_[0] - agent_pos[0]) + abs(self.wumpus_pos_[1] - agent_pos[1])<2:
            smell = 1
        if abs(self.hole_pos_[0] - agent_pos[0]) + abs(self.hole_pos_[1] - agent_pos[1])<2:
            breeze = 1
        return [smell,breeze]
    
    def nextState(self):
        a = self.agent.getAction()
        s = self.agent.getState()
        reward = self.DEFAULT_REWARD
        
        next_agent_pos = s[:2]
        n_flash = s[-1]
        if a < 5:
            next_agent_pos = self.moveAgent(s[:2], a)
        else:
            if n_flash > 0:
                n_flash -= 1
                flash_success = self.flashAgent(s, a)
                if flash_success:
                    reward += self.KILL_REWARD
            
        sense = self.updateSense(next_agent_pos)
        new_state = next_agent_pos+sense+[n_flash]
        (end_reward, end_flag) = self.testForEnd(new_state)

        if self.DYN_WUMPUS:
            self.moveWumpus()
        
        return (new_state, a, reward+end_reward, end_flag)
        
class WumpusHMI(Tk):

    def __init__(self, my_args=None):
        super().__init__()
        self.title("Wumpus world")
        self.DELTA_TIME = 1
        self.IMG_SIZE = 150
        self.LOGGER_TIME_STEP = (my_args["--verbose"]=="True")
        self.agent = Agent()
        self.reset()
        self.environment = Environment(self.agent,my_args)
        self.agent_prev_pos = self.agent.getPosition()
        self.wumpus_prev_pos = self.environment.getWumpusPosition()
        self.loadImages()
        self.createWorld()

    def loadImages(self):
        image = Image.open("./img/wumpus.gif")
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.ANTIALIAS)
        self.image_wumpus = ImageTk.PhotoImage(image)
        
        image = Image.open("./img/black.gif")
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.ANTIALIAS)
        self.image_hole = ImageTk.PhotoImage(image)
        
        image = Image.open("./img/papers.gif")
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.ANTIALIAS)
        self.image_treasure = ImageTk.PhotoImage(image)
        
        image = Image.open("./img/einstein.gif")
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.ANTIALIAS)
        self.image_hunter = ImageTk.PhotoImage(image)

    def convertCoord(self,pos):
        grid_size = self.environment.getGridSize()
        return (grid_size[1] - pos[1] - 1, pos[0])
        
    def createWorld(self):
        for line in range(self.environment.getGridSize()[0]):
            for col in range(self.environment.getGridSize()[1]):
                canvas = Canvas(self, width=self.IMG_SIZE, height=self.IMG_SIZE, bg='grey')
        
                if (line,col) == self.convertCoord(self.environment.getWumpusPosition()):
                    canvas.create_image(0, 0, anchor=NW, image=self.image_wumpus)

                if (line,col) == self.convertCoord(self.environment.getHolePosition()):
                    canvas.create_image(0, 0, anchor=NW, image=self.image_hole)

                if (line,col) == self.convertCoord(self.environment.getTreasurePosition()):
                    canvas.create_image(0, 0, anchor=NW, image=self.image_treasure)

                if (line,col) == self.convertCoord(self.agent.getPosition()):
                    canvas.create_image(0, 0, anchor=NW, image=self.image_hunter)
                canvas.grid(row=line, column=col)

    def updateWorld(self):
        for line in range(self.environment.getGridSize()[0]):
            for col in range(self.environment.getGridSize()[1]):
        
                if (line,col) == self.convertCoord(self.agent.getPosition()):
                    canvas = Canvas(self, width=self.IMG_SIZE, height=self.IMG_SIZE, bg='grey')
                    canvas.create_image(0, 0, anchor=NW, image=self.image_hunter)
                    canvas.grid(row=line, column=col)

                elif (line,col) == self.convertCoord(self.agent_prev_pos[:2]):
                    canvas = Canvas(self, width=self.IMG_SIZE, height=self.IMG_SIZE, bg='grey')
                    if (line,col) == self.convertCoord(self.environment.getWumpusPosition()):
                        canvas.create_image(0, 0, anchor=NW, image=self.image_wumpus)
                    elif (line,col) == self.convertCoord(self.environment.getHolePosition()):
                        canvas.create_image(0, 0, anchor=NW, image=self.image_hole)
                    elif (line,col) == self.convertCoord(self.environment.getTreasurePosition()):
                        canvas.create_image(0, 0, anchor=NW, image=self.image_treasure)                    
                    canvas.grid(row=line, column=col)

                if self.environment.DYN_WUMPUS or (self.wumpus_prev_pos[0] == -1 and self.environment.getWumpusPosition()[0] > -1):
                    if (line,col) == self.convertCoord(self.environment.getWumpusPosition()):
                        canvas = Canvas(self, width=self.IMG_SIZE, height=self.IMG_SIZE, bg='grey')
                        canvas.create_image(0, 0, anchor=NW, image=self.image_wumpus)
                        canvas.grid(row=line, column=col)

                    elif (line,col) == self.convertCoord(self.wumpus_prev_pos):
                        canvas = Canvas(self, width=self.IMG_SIZE, height=self.IMG_SIZE, bg='grey')
                        if (line,col) == self.convertCoord(self.environment.getHolePosition()):
                            canvas.create_image(0, 0, anchor=NW, image=self.image_hole)
                        elif (line,col) == self.convertCoord(self.environment.getTreasurePosition()):
                            canvas.create_image(0, 0, anchor=NW, image=self.image_treasure)                    
                        canvas.grid(row=line, column=col)
                    
    def reset(self):
        self.time_step_ = 0
        self.cumul_reward_ = 0
    
    def updateLoop(self):
        self.agent_prev_pos = self.agent.getPosition()
        self.wumpus_prev_pos = self.environment.getWumpusPosition()
        prev_state = self.agent.getState()
        (new_state, a, reward,end_flag) = self.environment.nextState()
        self.agent.nextState(new_state,reward)

        self.time_step_ += 1
        self.cumul_reward_ += reward
        if(self.LOGGER_TIME_STEP):
            print("time step " + str(self.time_step_) + " " + str(prev_state) + " " + str(a) + " " + str(self.agent.getState()) + " " + str(self.cumul_reward_))

        if(end_flag):
            print("End of episode at time step " + str(self.time_step_) + " " + str(self.cumul_reward_))
            self.reset()
            self.environment.reset()

        self.updateWorld()
        ihm.after(self.DELTA_TIME, self.updateLoop)

class RLPlatform:

    def __init__(self,my_args):
        self.LOGGER_TIME_STEP = (my_args["--verbose"]=="True")
        self.agent = Agent()
        self.reset()
        self.environment = Environment(self.agent,my_args)
        self.agent_prev_pos = self.agent.getPosition()

    def reset(self):
        self.time_step_ = 0
        self.cumul_reward_ = 0
    
    def updateLoop(self):
        self.agent_prev_pos = self.agent.getPosition()
        prev_state = self.agent.getState()
        (new_state, a, reward,end_flag) = self.environment.nextState()
        self.agent.nextState(new_state,reward)

        self.time_step_ += 1
        self.cumul_reward_ += reward
        if(self.LOGGER_TIME_STEP):
            print("time step " + str(self.time_step_) + " " + str(prev_state) + " " + str(a) + " " + str(self.agent.getState()) + " " + str(self.cumul_reward_))

        if(end_flag):
            print("End of episode at time step " + str(self.time_step_) + " " + str(self.cumul_reward_))
            self.reset()
            self.environment.reset()

if __name__ == "__main__":

    # Retrieve the arguments from the command-line
    my_args = docopt(__doc__)
    print(my_args)

    print(my_args["--hmi"])
    
    if my_args["--hmi"]=="True":
        ihm = WumpusHMI(my_args)
        ihm.updateLoop()
        ihm.mainloop()
    else:
        rl_platform = RLPlatform(my_args)
        for i in range(int(my_args["--max_n_episode"])):
            rl_platform.updateLoop()
