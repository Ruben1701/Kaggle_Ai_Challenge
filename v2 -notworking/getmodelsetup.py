from kaggle_environments import make
from lux.game import Game
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random

gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995
# we declare this global game_state object so that state persists across turns so we do not need to reinitialize it all the time
game_state = None

# this is the basic agent definition. At the moment this agent does nothing (and actually will survive for a bit before getting consumed by darkness)
def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    #         print("new episode")
    else:
        game_state._update(observation["updates"])

    actions = []

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    #     print(observation['reward'])

    return actions

# run another match but with our empty agent
env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
steps = env.run([agent, agent])
# print(f"steps: {steps}")

def get_inputs(game_state):
    # Teh shape of the map
    w,h = game_state.map.width, game_state.map.height
    # The map of ressources
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
    C = [ [[0, 0, 0, 0] for i in range(w)]  for j in range(h)]
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
# The shape of input
print(input_shape)

def get_model(game_state):
    inputs = keras.Input(shape=get_inputs(game_state).shape, name='The game map')

    # Try to play with the next layers in order to enhance the brean of your agent
    c = layers.Conv2D(8,(1,1),activation="relu")(inputs)
    c = layers.Conv2D(8,(1,1),activation="relu")(c)
    c = layers.Conv2D(8,(1,1),activation="relu")(c)

    # The next layer will define the direction among the 5 options, because we can take only one direction we had to use softmax activation
    output1 = layers.Dense(5, activation="softmax", name='direction')(c)

    # The next layer will define the option token by the unit (move, build,...), because one unit can take only one option we use sotmax activation
    output2 = layers.Dense(3, activation="softmax", name="option")(c)
    output = layers.concatenate([output1, output2])
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(loss='mse', optimizer="adam")
    return model

model = get_model(game_state)
print(model.summary())

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=1,
    show_dtype=1,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96)

x = get_inputs(game_state)
y = model.predict(np.asarray([x]))[0]

print(y.shape)

def get_prediction_actions(y,units):
    # move
    mv = np.argmax(y[:, :, :5], axis=2) # the index in this list  [c s n w e]

    choice = np.argmax(y[:, :, 5:], axis=2)
    actions = []
    for i in units:
        d = "csnwe"[mv[i.pos.y, i.pos.x]]
        if choice[i.pos.y, i.pos.x] == 0 and i.can_act():
            actions.append(i.move(d))
        elif choice[i.pos.y, i.pos.x] == 1 and i.can_build(game_state.map):
            actions.append(i.build_city())
        elif choice[i.pos.y, i.pos.x] == 2:
            actions.append(i.pillage())

    return actions,y[:, :, 5:]

print(get_prediction_actions(y,game_state.players[0].units)[0])

Last_State = {}

model = None
def agent(observation, configuration):
    global game_state,epsilon,model

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
        # Inistialise the model
        if not model:
            model= get_model(game_state)
            model.load_weights('../input/keras-lux-ai-weights/model.h5')

    else:
        game_state._update(observation["updates"])


    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    #     print(width, height)

    # Get Prediction of actions
    x = get_inputs(game_state)
    #     print(x.shape)
    y = model.predict(np.asarray([x]))[0]

    if random.random()<epsilon:
        y = np.random.rand(*y.shape)
    print("eps ", epsilon, end=" | ")
    actions, _ = get_prediction_actions(y,player.units)

    print("Reward",observation["reward"])
    # Model Learing

    if observation.player in Last_State:
        _x, _y, _ = Last_State[observation.player]
        state, next_state, reward = _x, x, observation["reward"]
        #         print(_y.shape,_.shape)
        #         reward = sigmoid(reward)
        reward /= 100000.

        reward = reward+ gamma * np.amax(_y,axis = 2)
        # print(reward.shape)

        for i in player.units:
            print(_y[i.pos.x,i.pos.y].shape,_[i.pos.x,i.pos.y].shape)
            _y[i.pos.y,i.pos.x][[0,1,2,3,4]] = reward[i.pos.x,i.pos.y]
            _y[i.pos.y,i.pos.x][5] = reward[i.pos.y,i.pos.x] if reward[i.pos.y,i.pos.x]>=0.5 else 1-reward[i.pos.y,i.pos.x]
            _y[i.pos.y,i.pos.x][6] = reward[i.pos.y,i.pos.x] if reward[i.pos.y,i.pos.x]>=0.5 else 1-reward[i.pos.y,i.pos.x]
            _y[i.pos.y,i.pos.x][7] = reward[i.pos.y,i.pos.x] if reward[i.pos.y,i.pos.x]>=0.5 else 1-reward[i.pos.y,i.pos.x]
        model.fit(np.asarray([state]),np.asarray([_y]), epochs=1, verbose=1)
        if epsilon > epsilon_final:
            epsilon*= epsilon_decay
    Last_State[observation.player] = [x,y,_]
    return actions

episodes = input("Enter amount of episodes")

for eps in range(int(episodes)):
    print("=== Episode {} ===".format(eps))
    env = make("lux_ai_2021", configuration={"seed": 562124210, "loglevel": 2}, debug=True)
    steps = env.run(["getmodelsetup.py", agent])

model.save_weights("model.h5")