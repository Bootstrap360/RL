import matplotlib.pyplot as plt
from environments.gridworld2 import gameEnv
import random

plt.subplot('511')
env = gameEnv(partial=False,size=10)

state, reqard, done = env.step(1)
plt.subplot('512')
plt.imshow(state)

state, reqard, done = env.step(1)
plt.subplot('513')
plt.imshow(state)

state, reqard, done = env.step(1)
plt.subplot('514')
plt.imshow(state)

state, reqard, done = env.step(1)
plt.subplot('515')
plt.imshow(state)
plt.show()

while True:
    state, reqard, done = env.step(random.randint(0, gameEnv.get_num_actions()))
