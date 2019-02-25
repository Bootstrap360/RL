
import environments.gridworld2

def make(env_name):
    if env_name == 'grid_world2':
        return gridworld2.GameEnv(partial=False,size=10)