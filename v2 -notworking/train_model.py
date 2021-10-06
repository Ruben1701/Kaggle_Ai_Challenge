from kaggle_environments import make
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES, Position
from lux.game_objects import Unit
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math, sys
import numpy as np
import random
from IPython.display import clear_output
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_hub as hub
from collections import deque
import random
import math

episodes = input("Enter amount of episodes")

for eps in range(episodes):
    clear_output()
    print("=== Episode {} ===".format(eps))
    env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
    steps = env.run(["../getmodelsetup.py", agent])

# clear_output()