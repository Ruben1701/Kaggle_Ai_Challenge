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

# from pathlib import Path
# p = Path('/kaggle_simulations/agent/')
# if p.exists():
#     sys.path.append(str(p))
# else:
#     p = Path('__file__').resolve().parent


game_state = None
model = None
epsilon = None

def get_inputs(game_state, observation):

    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]

    # The shape of the map
    w,h = game_state.map.width, game_state.map.height
    # The map of resources
    M = [ [0  if game_state.map.map[j][i].resource==None else game_state.map.map[j][i].resource.amount for i in range(w)]  for j in range(h)]

    M = np.array(M).reshape((h,w,1))

    # The map of units features
    U_player = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = player.units
    for i in units:
        U_player[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]
    U_player = np.array(U_player)

    U_opponent = [ [[0,0,0,0,0] for i in range(w)]  for j in range(h)]
    units = opponent.units
    for i in units:
        U_opponent[i.pos.y][i.pos.x] = [i.type,i.cooldown,i.cargo.wood,i.cargo.coal,i.cargo.uranium]

    U_opponent = np.array(U_opponent)

    # The map of cities featrues
    e = player.cities
    C_player = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_player[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_player = np.array(C_player)

    e = opponent.cities
    C_opponent = [ [[0,0,0] for i in range(w)]  for j in range(h)]
    for k in e:
        citytiles = e[k].citytiles
        for i in citytiles:
            C_opponent[i.pos.y][i.pos.x] = [i.cooldown,e[k].fuel,e[k].light_upkeep]
    C_opponent = np.array(C_opponent)

    # stacking all in one array
    E = np.dstack([M,U_opponent,U_player,C_opponent,C_player])
    return E


def get_model(s, observation, game_state):
    input_shape = (s,s,17)
    inputs = keras.Input(shape= input_shape,name = 'The game map')
    f = layers.Flatten()(inputs)
    h,w,_ = get_inputs(game_state, observation).shape
    print(h,w)
    #     output = layers.Dense(w*h*8,activation = "sigmoid")(f)

    f = layers.Dense(w*h,activation = "sigmoid")(f)
    f = layers.Reshape((h,w,-1))(f)
    units = layers.Dense(6,activation = "softmax",name = "Units_actions")(f)

    cities = layers.Dense(2,activation = "sigmoid",name = "Cities_actions")(f)

    output = layers.Concatenate()([units,cities])
    model = keras.Model(inputs = inputs, outputs = output)
    return model



def get_prediction_actions(y,player, game_state):
    # move
    option = np.argmax(y,axis = 2)
    # c s n w e build_city & research & buid_worker
    actions = []
    for i in player.units:
        #         print(option.shape,i.pos.y,i.pos.x)
        d = "csnwe#############"[option[i.pos.y,i.pos.x]]
        if option[i.pos.y,i.pos.x]<5:actions.append(i.move(d))
        elif option[i.pos.y,i.pos.x]==5 and i.can_build(game_state.map):actions.append(i.build_city())

    for city in player.cities.values():
        for city_tile in city.citytiles:
            if option[city_tile.pos.y,city_tile.pos.x]==6:
                action = city_tile.research()
                actions.append(action)
            if option[city_tile.pos.y,city_tile.pos.x]==7:
                action = city_tile.build_worker()
                actions.append(action)
    return actions,option

def agent(observation, configuration):
    global game_state
    global epsilon
    global model

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        print("Creating model..")
        model = get_model(game_state.map.width, observation, game_state)
        print("Load model weight..")
        try:
            model.load_weights( str('./model16test.h5'%game_state.map.width),  by_name=True, skip_mismatch=True)
        except Exception as e:
            print('Error in model load')
            print(e)
        #         model = tf.keras.models.load_model('model.h5')
        print("Done creating mdoel")


    else:
        game_state._update(observation["updates"])


    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state, observation)
    y = model.predict(np.asarray([x]))[0]
    actions, _ = get_prediction_actions(y, player, game_state)
    return actions