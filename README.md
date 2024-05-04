# rl-implementations
Repo where I practice implementing different RL algorithms

# Common Problems
## 1. Rendering 
To resolve these problems, refer to the Rendering section in the [DM Control repo on GitHub](https://github.com/google-deepmind/dm_control)

Another common problem is that the MuJoCo Python bindings need to be set. Make sure you 
```python
import os 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl' # Can also be 'glfw' or 'osmesa'

# Main imports 
from dm_control import suite
...
```