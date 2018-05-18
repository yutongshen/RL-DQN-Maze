from src.rl import RL
from src.enviroment import Env
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        default='4',
                        dest='LENGTH',
                        help='input the length of the grid')

    parser.add_argument('-i',
                        default='200',
                        dest='ITERATION',
                        help='input the iteration of training')

    parser.add_argument('-m',
                        default='2000',
                        dest='MEMORYSIZE',
                        help='input the size of memory')

    parser.add_argument('-d',
                        default='.1',
                        dest='DELAY',
                        help='input delay')

    args = parser.parse_args()
    
    try:
        length    = int(args.LENGTH)
        iteration = int(args.ITERATION)
        mem_size  = int(args.MEMORYSIZE)
    except ValueError:
        print('error: length, iteration or memory size must be an integer')
        sys.exit()

    try:
        delay = float(args.DELAY)
    except ValueError:
        print('error: delay must be an float')
        sys.exit()

    game = Env(length)
    rl   = RL(2, game.get_actions(), Memory_size=mem_size)

    step = 0
    while game.episode < iteration:
        s, done = game.restart()
        while not done:
            a = rl.actor(s)
            ns, r, done = game.go(a)
            rl.store_observation(s, a, r, ns)
            if step > mem_size and step % 5 == 0:
                rl.learn()
            s = ns
            step += 1

    print()
    for i in range(-2, 3):
        for j in range(-2, 3):
            print('(', i, j, ')', rl.q_evaluation_model.predict(np.array([[i, j]]))[0])


    plt.plot(np.arange(len(game.history)), game.history)
    plt.show()
    plt.plot(np.arange(len(rl.history)), rl.history)
    plt.show()
