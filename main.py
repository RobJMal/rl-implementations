import os 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np 

if __name__ == '__main__':
    random_state = np.random.RandomState(42)    # Setting the seed 
    domain_name, task_name = 'cartpole', 'swingup'
    env = suite.load(domain_name='cartpole', 
                     task_name='swingup', 
                     task_kwargs={'random': random_state})
    pixels = env.physics.render()

    plt.imshow(pixels)
    plt.axis('off')
    plt.show()