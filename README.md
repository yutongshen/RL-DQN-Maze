# Reinforce Learning Practice
Maze - Deep Q Network

## Prerequisite
- Python 3.6.4

## Install Dependency
```sh
$ pip install -r requirements.txt
```

## Usage
```sh
$ python usage: main.py [-h] [-l LENGTH] [-i ITERATION] [-m MEMORYSIZE] [-d Delay]
```

| optional Options           | Description                                    |
| ---                        | ---                                            |
| -h, --help                 | show this help message and exit                |
| -l LENGTH                  | input the length of the grid                   |
| -i ITERATION               | input the iteration of training                |
| -m MEMORYSIZE              | input the size of memory                       |
| -d DELAY                   | input delay                                    |

## Game Rules

\_ \_ \_ \* ![\leftarrow](https://latex.codecogs.com/svg.latex?\leftarrow) Destination, will get +1 reward <br>
\_ X \_ \_<br>
\_ \_ \_ \_<br>
o \_ \_ \_<br>
![\uparrow](https://latex.codecogs.com/svg.latex?\uparrow)<br>
Start Point<br>

We can choose 'up', 'down', 'left' and 'right' to approach destination

- \*: Destination
- o: Start
- X: Trap

## Algorithm
- Deep Q Network
  - Initialize Q network with parameters θ
  - Initialize enviroment and get current state s
  - According to s, Actor will give an action a: (ε-Greedy, e.g. ε = 0.9)
    - 10%: random choose one of 'up', 'down', 'left' or 'right'
    - 90%: choose the action with the highest ![Q(s:\theta)](https://latex.codecogs.com/svg.latex?Q%28s;\theta%29)
  - Take the action, and observe the reward, r, as well as the new state, s'.
  - Update the θ for the state using the observed reward and the maximum reward possible for the next state.
    - ![loss=(r+\gamma\max\_{a'}Q(s',a';\theta^{-})-Q(s,a:\theta))^{2}](https://latex.codecogs.com/svg.latex?loss=%28r+\gamma%20max_{a%27}Q%28s%27,a%27;\theta^{-}%29-Q%28s,a;\theta%29%29^{2})
    - ![\theta=\theta-lr\triangledown\theta](https://latex.codecogs.com/svg.latex?\theta=\theta-lr\triangledown\theta)
  - Every C steps reset ![\theta^{-}\leftarrow\theta](https://latex.codecogs.com/svg.latex?\theta^{-}\leftarrow\theta)
  - Set the state to the new state, and repeat the process until a terminal state is reached.

## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)
