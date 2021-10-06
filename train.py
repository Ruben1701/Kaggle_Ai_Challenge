import random

import math, sys
from IPython.core.display import clear_output
from kaggle_environments import make

from lux.game import Game
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

from agent import get_inputs, get_prediction_actions
from model import get_model
from v1.old_agent import agent as old_agent

Last_State = {}
learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
game_state = None
model = None
last_reward = 0
W = 0

def agent(observation, configuration):
    global game_state,epsilon,model,last_reward,W

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])


    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Get Prediction of actions
    x = get_inputs(game_state, observation)
    y = model.predict(np.asarray([x]))[0]

    if random.random()<epsilon:
        y = np.random.rand(*y.shape)
    print("eps ",epsilon,end= " | ")
    actions,option = get_prediction_actions(y,player, game_state)

    print("Reward",observation["reward"])


    if observation.player in Last_State:
        _x,_y,_player,_option = Last_State[observation.player]
        state,next_state,reward = _x,x,observation["reward"]

        # Reward
        if reward > last_reward:r=1
        elif reward < last_reward:r = -1
        else:r = 0

        # Q-learning update

        for i in _player.units:
            Q1 = _y[i.pos.y,i.pos.x][_option[i.pos.y,i.pos.x]]
            Q2 = y[i.pos.y,i.pos.x][_option[i.pos.y,i.pos.x]]
            v = r + gamma*(Q2 - Q1)
            _y[i.pos.y,i.pos.x][_option[i.pos.y,i.pos.x]] += learning_rate*v

        _y = y + learning_rate*_y

        states = [state]
        _y_ = [_y]

        model.fit(np.asarray(states),np.asarray(_y_), epochs=1, verbose=1)
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
    Last_State[observation.player] = [x, y, player, option]
    last_reward = observation["reward"]
    return actions

episodes = input("Input amount of episodes: ")

# RL training
# sizes = [12,16,24,32]
sizes = [32]

for size in sizes:
    # Initialise the model
    model = get_model(size)
    Last_State = {}
    for eps in range(int(episodes)):
        epsilon = 0.2 # Maintaining exploration
        clear_output()
        print("=== Episode {} ===".format(eps))
        env = make("lux_ai_2021", configuration={"annotations": True, "width":size, "height":size}, debug=True)
        steps = env.run([agent, old_agent])
    # Save the model
    model.save_weights("modelll_%d.h5"%size)