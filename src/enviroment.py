import numpy as np
import time

class Env:
    def __init__(self, length = 4, delay = .1):
        self.state  = None
        self.length = length
        self.delay  = delay
        f = lambda: np.random.choice(range(length))
        point = []
        while len(point) < 3:
            p = f(), f()
            if p in point:
                continue
            point.append(p)
        self.start   = point[0]
        self.trap    = point[1]
        self.dest    = point[2]
        # self.start   = (0, 0)
        # self.trap    = (1, 2)
        # self.dest    = (2, 2)
        self.episode = 0
        self.step    = 0
        self.total_r = 0
        self.history = []

    def get_actions(self):
        return ['up', 'down', 'left', 'right']

    def render(self):
        view = np.array([['_ '] * self.length] * self.length)
        view[self.dest]  = '* '
        view[self.trap]  = 'X '
        view[self.state] = 'o '
        interaction = '  '
        for v in view:
            interaction += ''.join(v) + '\n  '
        message = 'EPISODE: {}, STEP: {}, REWARD: {}'.format(self.episode, self.step, self.total_r) 
        interaction += message
        print(interaction)
        # time.sleep(self.delay)

    def go(self, action):
        done = False
        reward = 0.

        f = {
            'up':    lambda s: (s[0] - 1, s[1]),
            'down':  lambda s: (s[0] + 1, s[1]),
            'left':  lambda s: (s[0], s[1] - 1),
            'right': lambda s: (s[0], s[1] + 1)
        }.get(action, lambda s: s)
        
        print()
        print('  {}'.format(action))

        self.state = f(self.state)
        if self.state[0] == -1:
             # reward = -.1
             self.state = 0, self.state[1]
        elif self.state[0] == self.length:
             # reward = -.1
             self.state = self.length - 1, self.state[1]
        elif self.state[1] == -1:
             # reward = -.1
             self.state = self.state[0], 0
        elif self.state[1] == self.length:
             # reward = -.1
             self.state = self.state[0], self.length - 1

        if self.state == self.dest:
            reward = 1.
            done   = True
        elif self.state == self.trap:
            reward = -1.
            done   = True
        
        reward -= .1
        self.step += 1
        self.total_r += reward

        self.render()

        if done:
            self.history.append(self.total_r)
            self.episode += 1
            self.step = 0
            self.total_r = 0

        return np.array(self.state) - np.array(self.dest), reward, done 


    def restart(self):
        self.state = self.start
        self.render()

        return np.array(self.state) - np.array(self.dest), False
