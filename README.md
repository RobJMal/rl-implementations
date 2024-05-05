# rl-implementations
Repo where I practice implementing different RL algorithms

# Common Problems
## 1. Rendering 
To resolve these problems, refer to the Rendering section in the [DM Control repo on GitHub](https://github.com/google-deepmind/dm_control)

Another common problem is that the MuJoCo Python bindings need to be set. Make sure you import the `os` library and then set the graphics library environment BEFORE importing `dm_control`. 
```python
import os 

# Sets the graphics library for MuJoCo, need to set this 
# before importing dm_control 
os.environ['MUJOCO_GL'] = 'egl' # Can also be 'glfw' or 'osmesa'

# Main imports 
from dm_control import suite
...
```

## 2. Rendering with `gymnasium`
When using a conda environment, run the following command: `conda install -c conda-forge libstdcxx-ng`. This updates the lbraries related to visualization to a higher version. 

Source: https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found 