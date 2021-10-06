import logging
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

def get_inputs(game_state):
    # The shape of the map
    w,h = game_state.map.width, game_state.map.height
    # The map of resources
    M = [ [0  if game_state.map.map[j][i].resource==None else game_state.map.map[j][i].resource.amount for i in range(w)]  for j in range(h)]

    M = np.array(M).reshape((h,w,1))

    # The map of units features
    U = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = game_state.players[0].units
    for i in units:
        U[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]

    U = np.array(U)

    # The map of cities featrues
    e = game_state.players[1].cities
    C = [ [[0,0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep,e[k].team]

    C = np.array(C)
    #     print(M.shape,U.shape,C.shape)
    # stacking all in one array
    E = np.dstack([M,U,C])
    return E

input_shape = get_inputs(game_state).shape
input_shape