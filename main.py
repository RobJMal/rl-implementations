import os 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'

# Main imports 
from dm_control import suite
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = suite.load('cartpole', 'swingup')
    pixels = env.physics.render()

    plt.imshow(pixels)
    plt.axis('off')
    plt.show()