import numpy as np
import random, copy, math
import itertools
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class gameOb():
    def __init__(self,coordinates,size,color,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.name = name
        self.reward = reward
    
class Hero(gameOb):

    def __init__(self, coordinates, size, sizeX, sizeY):
        self.color = [255,255,100]
        super(Hero, self).__init__(coordinates, size, self.color, None,'hero')
        self.sizeX = sizeX
        self.sizeY = sizeY


    def take_action(self, action):
        if action == 0 and self.y >= 1:
            self.y -= 1
        if action == 1 and self.y < self.sizeY - 2:
            self.y += 1
        if action == 2 and self.x >= 1:
            self.x -= 1
        if action == 3 and self.x < self.sizeX - 2:
            self.x += 1
    
    @staticmethod
    def get_num_actions():
        return 4

class Goal(gameOb):
    def __init__(self,coordinates,size):
        self.color = [0,255,0]
        super(Goal, self).__init__(coordinates,size, self.color,1,'goal')

class Fire(gameOb):
    def __init__(self,coordinates,size):
        self.color = [255,0,0]
        super(Fire, self).__init__(coordinates,size, self.color,-11,'fire')

class Environment():
    def __init__(self, height, width, channels, partial = False):
        self.partial = partial
        self.height = height
        self.width = width
        self.channels = channels
        self.reset()

    def update(self, objects):
        self.observation_space = np.zeros([self.height,self.width, self.channels], dtype=int)
        hero = None
        for item in objects:
            for i, val in enumerate(item.color):
                self.observation_space[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,i] = val
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            self.observation_space = self.observation_space[hero.y:hero.y+3,hero.x:hero.x+3,:]
    
    def reset(self):
        if self.partial:
            self.observation_space = np.zeros([3, 3, 3], dtype=int)
        else:
            self.observation_space = np.zeros([self.height,self.width, self.channels], dtype=int)
    
    def get_empty(self):
        return np.zeros([self.height,self.width, self.channels], dtype=int)

class GameEnv():
    def __init__(self, partial, size, state_type = 'image'):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.height = 10
        self.width = 10
        self.channels = 3
        self.state_type = state_type
        self.min_distance = 1000000
        self.env = Environment(size, size, 3)

        a = self.reset()
        plt.imshow(a,interpolation="nearest")
    
    def get_num_states(self):
        return self.height * self.width * self.channels
    
    @staticmethod
    def get_num_actions():
        return Hero.get_num_actions()
    
    def get_state(self):
        self.env.update(self.objects)
        return self.env.observation_space
        
    def reset(self):
        self.objects = []
        hero = Hero(self.newPosition(),1, self.width, self.height)
        self.objects.append(hero)

        for i in range(1):
            goal = Goal(self.newPosition(),1)
            self.objects.append(goal)
        for i in range(0):
            fire = Fire(self.newPosition(),1)
            self.objects.append(fire)

        return self.get_state()

    def take_action(self,action):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.0
        hero.take_action(action)
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def newPosition(self):
        iterables = [ range(self.sizeX-1), range(self.sizeY-1)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]
    
    @staticmethod
    def get_distace(target, hero):
        return math.pow(target.x - hero.x, 2) + math.pow(target.y - hero.y, 2)

    def check_goal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        target = others[0]
        distance = GameEnv.get_distace(target, hero)
        reward = 0
        if distance < self.min_distance and distance != 0:
            reward = 1.0 / distance
            self.min_distance = distance
        if hero.x == target.x and hero.y == target.y:
            self.objects.remove(target)
            self.objects.append(Goal(self.newPosition(),1))
            self.min_distance = 100000000

            return 100, True
        return reward, False
        # for other in others:
        #     if hero.x == other.x and hero.y == other.y:
        #         self.objects.remove(other)
        #         if other.reward == 1:
        #             self.objects.append(Goal(self.newPosition(),1))
        #         # else: 
        #         #     self.objects.append(gameOb(self.newPosition(),1,1,self.colors['fire'],-1,'fire'))
        #         return 1, True
        if ended == False:
            return 0.0,False

    def render_env(self):
        a = copy.deepcopy(self.env.observation_space)
        render_height = 128 * 2
        render_width = 128 * 2
        b = scipy.misc.imresize(a[:,:,0],[render_width, render_height, 1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[render_width, render_height, 1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[render_width, render_height, 1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a
    
    def render(self):
        rendered_env = self.render_env()
        cv2.imshow('game', rendered_env)
        cv2.waitKey(10)


    def step(self,action):
        penalty = self.take_action(action)
        reward, done = self.check_goal()
        state = self.get_state()
        info = 'I dont know'
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done, info
        else:
            return state,(reward+penalty),done, info