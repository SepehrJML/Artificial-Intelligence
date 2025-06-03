from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np

import matplotlib.pyplot as plt
class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = dict()


        self.lr = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.hist_reward = []
        
        self.temp = 0
        self.something = 0
    def get_optimal_policy(self, state):
        if state not in self.q_table.keys():
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    def epsilon_decay(self):
        if self.epsilon > 0.01:
            self.epsilon *= 0.999
        return self.epsilon   
    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon_decay():
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
            self.q_table[state][state[1]] = 1
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
            self.q_table[next_state][next_state[1]] = 1
        self.q_table[state][action] = (1 - self.lr) * (self.q_table[state][action]) + self.lr * (reward + self.discount_factor * np.max(self.q_table[next_state]))
    def distance_from_apple(self, snack):
        if ( abs(self.head.pos[0] - snack.pos[0]) > abs(self.head.pos[1] - snack.pos[1]) ) and (self.head.pos[0] > snack.pos[0] ):
            return 0 #Left
        elif ( abs(self.head.pos[0] - snack.pos[0]) > abs(self.head.pos[1] - snack.pos[1]) ) and (self.head.pos[0] <= snack.pos[0]):
            return 1 #Right
        elif ( abs(self.head.pos[0] - snack.pos[0]) <= abs(self.head.pos[1] - snack.pos[1]) ) and (self.head.pos[1] > snack.pos[1] ):
            return 2 #Down
        elif ( abs(self.head.pos[0] - snack.pos[0]) <= abs(self.head.pos[1] - snack.pos[1]) ) and (self.head.pos[1] <= snack.pos[1]):
            return 3 #Up

        
    def ob_space(self,other_snake):
        
        directions = np.array([-2,-1,0,1,2])
        snake_body = set(map(lambda z: z.pos, self.body))
        other_snake_body = set(map(lambda z: z.pos, other_snake.body))
        output = []
        output = [0 if (x < 1 or y < 1 or x >= ROWS - 1 or  y >= ROWS - 1 or (x, y) in snake_body or ( (x, y) in other_snake_body or (x, y) == other_snake.head.pos) )
                  else 1 for x in directions + self.head.pos[0] for y in directions + self.head.pos[1]]
        return tuple(output)
    
    def make_state(self, snack, other_snake):
        space = self.ob_space(other_snake)
        move_to_apple = self.distance_from_apple(snack)
        return space,move_to_apple
        
    def move(self, snack, other_snake):
        state = self.make_state(snack, other_snake)
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        new_state = self.make_state(snack, other_snake)
        return state, new_state, action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board
            reward += BORDER_PUNISH
            win_other = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            # TODO: Reward the snake for eating
            reward += EAT_REWARD
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself
            reward += HIT_PUNISH
            win_other = True
            reset(self, other_snake)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake
                reward += HIT_PUNISH
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    # TODO: Reward the snake for hitting the head of the other snake and being longer
                    reward += HEAD_REWARD
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    # TODO: No winner
                    pass
                else:
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter
                    reward += HEAD_PUNISH
                    win_other = True
                    
            reset(self, other_snake)
        self.temp += reward
        if (self.something % 1 == 0):
            self.hist_reward.append(self.temp)
        self.something += 1
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        plt.plot(self.hist_reward)
        plt.show()
        np.save(file_name, self.q_table)
        