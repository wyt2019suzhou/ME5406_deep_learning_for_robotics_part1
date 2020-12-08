#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tkinter as tk
import random
from itertools import product

"""
ENVIRONMENT

Red circle:          robot.
Black rectangles:       hole       [reward = -1]
Green circle:      frisbee    [reward = +1] 
All other states:       ice      [reward = 0]
"""
# Each grid's pixels
UNIT = 40

random.seed(715)


class frozenlake(tk.Tk, object):
    def __init__(self, name='4*4', slide=False, slide_p=0.5):
        """choose model and build initial environment"""
        super(frozenlake, self).__init__()
        if name != '4*4' and name != '10*10':
            raise ValueError('Must be 4*4 or 10*10')
        self.slide = slide
        self.slide_p = slide_p
        # all action
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # choose model
        self.name = name
        if self.name == '4*4':
            self.grid_num = 4
            self.hole_num = 4
            center = [[c, r] for c, r in product([20, 60, 100, 140], [20, 60, 100, 140])]
        if self.name == '10*10':
            self.grid_num = 10
            self.hole_num = 25
            center = [[c, r] for c, r in product([20, 60, 100, 140, 180, 220, 260, 300, 340, 380],
                                                 [20, 60, 100, 140, 180, 220, 260, 300, 340, 380])]
        self.n_observation = self.grid_num * self.grid_num
        # all state
        self.observation_space = []
        for i in range(self.n_observation):
            self.observation_space.append(
                [center[i][0] - 20.0, center[i][1] - 20.0, center[i][0] + 20.0, center[i][1] + 20.0])
        self.title('frozen_lake')
        self.geometry('{0}x{1}'.format(self.grid_num * UNIT, self.grid_num * UNIT))
        # build initial environment
        self._build_env()
        self.hole_location = []
        for i in range(self.hole_num):
            self.hole_location.append(self.canvas.coords(self.hole[i]))

    def _build_env(self):
        """build initial environment"""
        # create canvas
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.grid_num * UNIT,
                                width=self.grid_num * UNIT)

        # create grids
        for c in range(0, self.grid_num * UNIT, UNIT):
            # draw Vertical line
            x0, y0, x1, y1 = c, 0, c, self.grid_num * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.grid_num * UNIT, UNIT):
            # draw Horizontal line
            x0, y0, x1, y1 = 0, r, self.grid_num * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # create hole
        hole_center = {j: np.array([0, 0]) for j in range(self.grid_num)}
        self.hole = {j: 0 for j in range(self.grid_num)}

        if self.name == '4*4':
            hole_center[0] = origin + np.array([UNIT, UNIT])
            self.hole[0] = self.canvas.create_rectangle(
                hole_center[0][0] - 20, hole_center[0][1] - 20,
                hole_center[0][0] + 20, hole_center[0][1] + 20,
                fill='black')

            hole_center[1] = origin + np.array([UNIT * 3, UNIT])
            self.hole[1] = self.canvas.create_rectangle(
                hole_center[1][0] - 20, hole_center[1][1] - 20,
                hole_center[1][0] + 20, hole_center[1][1] + 20,
                fill='black')

            hole_center[2] = origin + np.array([UNIT * 3, UNIT * 2])
            self.hole[2] = self.canvas.create_rectangle(
                hole_center[2][0] - 20, hole_center[2][1] - 20,
                hole_center[2][0] + 20, hole_center[2][1] + 20,
                fill='black')

            hole_center[3] = origin + np.array([0, UNIT * 3])
            self.hole[3] = self.canvas.create_rectangle(
                hole_center[3][0] - 20, hole_center[3][1] - 20,
                hole_center[3][0] + 20, hole_center[3][1] + 20,
                fill='black')

        # Distribute the holes randomly for 10*10 environment
        if self.name == '10*10':
            more_pos = []
            location = [[c, r] for c, r in product(range(10), range(10))]
            while len(more_pos) < 25:
                one_pos = location[random.randint(0, 99)]
                if one_pos != [0, 0] and one_pos != [9, 9] and one_pos not in more_pos:
                    more_pos.append(one_pos)
            for i in range(25):
                hole_center[i] = origin + np.array([UNIT * more_pos[i][0], UNIT * more_pos[i][1]])
                self.hole[i] = self.canvas.create_rectangle(
                    hole_center[i][0] - 20, hole_center[i][1] - 20,
                    hole_center[i][0] + 20, hole_center[i][1] + 20,
                    fill='black')

        # create frisbee
        frisbee_center = origin + np.array([UNIT * (self.grid_num - 1), UNIT * (self.grid_num - 1)])
        self.frisbee = self.canvas.create_oval(
            frisbee_center[0] - 20, frisbee_center[1] - 20,
            frisbee_center[0] + 20, frisbee_center[1] + 20,
            fill='green')

        # create robot
        self.robot = self.canvas.create_oval(
            origin[0] - 20, origin[1] - 20,
            origin[0] + 20, origin[1] + 20,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        """Robot back to origin and return it's coordinate"""
        self.update()
        self.canvas.delete(self.robot)
        origin = np.array([20, 20])
        self.robot = self.canvas.create_oval(
            origin[0] - 20, origin[1] - 20,
            origin[0] + 20, origin[1] + 20,
            fill='red')
        state = self.canvas.coords(self.robot)
        state = self.observation_space.index(state)
        return state

    def step(self, action):
        """input action return new state, reward, done"""
        s = self.canvas.coords(self.robot)
        base_action = np.array([0, 0])
        # if choose slide environment ,robot will has a certain probability to perform other actions
        if self.slide and np.random.uniform() < self.slide_p:
            action = np.random.choice([0, 1, 2, 3])
        # apply action and restrict it not out area
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (self.grid_num - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (self.grid_num - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT
        # move robot
        self.canvas.move(self.robot, base_action[0], base_action[1])
        # next state
        next_state = self.canvas.coords(self.robot)

        # reward function
        if next_state == self.canvas.coords(self.frisbee):
            reward = 1
            done = True

        elif next_state in self.hole_location:
            reward = -1
            done = True

        else:
            reward = 0
            done = False
        # use the index of grids as state
        next_state = self.observation_space.index(next_state)

        return next_state, reward, done

    def validaction(self):
        """return the actions that can be performed in the existing state"""
        s = self.canvas.coords(self.robot)
        s = self.observation_space.index(s)
        if self.name == '4*4':
            if s == 5 or s == 6 or s == 9 or s == 10:
                valid_a = [0, 1, 2, 3]
            elif s == 0:
                valid_a = [1, 2]
            elif s == 4 or s == 8:
                valid_a = [1, 2, 3]
            elif s == 12:
                valid_a = [1, 3]
            elif s == 1 or s == 2:
                valid_a = [0, 1, 2]
            elif s == 13 or s == 14:
                valid_a = [0, 1, 3]
            elif s == 3:
                valid_a = [0, 2]
            elif s == 15:
                valid_a = [0, 3]
            elif s == 7 or s == 11:
                valid_a = [0, 2, 3]
        if self.name == '10*10':
            if s == 0:
                valid_a = [1, 2]
            elif s in range(1, 9):
                valid_a = [0, 1, 2]
            elif s == 9:
                valid_a = [0, 2]
            elif s % 10 == 0 and s != 0 and s != 90:
                valid_a = [1, 2, 3]
            elif s % 10 == 9 and s != 9 and s != 99:
                valid_a = [0, 2, 3]
            elif s == 90:
                valid_a = [1, 3]
            elif s == 99:
                valid_a = [0, 3]
            else:
                valid_a = [0, 1, 2, 3]
        return valid_a

    def render(self):
        """out put a picture"""
        self.update()
