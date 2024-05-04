import os 
import copy 
import itertools
from IPython.display import clear_output

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'    # Connecting to GPU

# Main imports 
from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np 

from visualization import save_video

if __name__ == '__main__':
    random_state = np.random.RandomState(42)    # Setting the seed 
    dmc_domain_name, dmc_task_name = 'cartpole', 'swingup'
    env = suite.load(domain_name=dmc_domain_name, 
                     task_name=dmc_task_name, 
                     task_kwargs={'random': random_state})
    pixels = env.physics.render()

    duration = 4
    frames = []
    ticks = []
    rewards = []
    observations = []

    spec = env.action_spec()    # Environment specifications (range and shape of valid actions)
    time_step = env.reset()

    while env.physics.data.time < duration: 
        action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        time_step = env.step(action)

        # Frames of the agent in the environment (for visualization purposes)
        camera0 = env.physics.render(camera_id=0, height=200, width=200)
        camera1 = env.physics.render(camera_id=1, height=200, width=200)
        frames.append(np.hstack((camera0, camera1)))

        rewards.append(time_step.reward)
        observations.append(copy.deepcopy(time_step.observation))
        ticks.append(env.physics.data.time)

    output_policy_video_filename = f'media/{dmc_domain_name}-{dmc_task_name}_result.mp4'
    save_video(frames=frames, filename=output_policy_video_filename, framerate=30)