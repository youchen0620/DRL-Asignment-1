import numpy as np
import pickle
import random
import gym
import time

try:
    with open("qtable.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("Loaded Q-table from qtable.pkl")
except FileNotFoundError:
    q_table = {}

def get_state(obs, target_pos, has_picked_up):
    agent_pos = (obs[0], obs[1])
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs[10], obs[11], obs[12], obs[13], obs[14], obs[15]
    return (agent_pos, target_pos, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look, has_picked_up)

def calculate_manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

initialized = False
previous_state = None
previous_action = None
inference_step = 0

def get_action(obs):
    global initialized
    global previous_state
    global previous_action
    global inference_step

    stations = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]

    if not initialized or (previous_state != None and calculate_manhattan_distance(previous_state[0], (obs[0], obs[1])) > 1):
        initialized = True
        inference_step = 0

        target_pos = stations[0]
        for i in range(1, len(stations)):
            prev_dist_to_target = calculate_manhattan_distance((obs[0], obs[1]), target_pos)
            curr_dist_to_target = calculate_manhattan_distance((obs[0], obs[1]), stations[i])

            if curr_dist_to_target < prev_dist_to_target:
                target_pos = stations[i]

        state = get_state(obs, target_pos, False)
        inference_step += 1

        action = None
        if state not in q_table:
            action = random.choice(range(6))
        else:
            action_probs = softmax(q_table[state])
            action = int(np.random.choice(len(action_probs), p=action_probs))
        
        previous_state = state
        previous_action = action

        return action
    
    state = get_state(obs, previous_state[1], previous_state[8])
    inference_step += 1

    if (state[0] == state[1] and ((not state[6] or (not state[7] and state[8])) or (state[7] and not state[8]))) or inference_step >= 30:
        state = get_state(obs, stations[(stations.index(state[1]) + 1) % 4], state[8])
        inference_step = 0
    if previous_state[0] == state[0] and state[0] == state[1] and state[6] and not state[8] and previous_action == 4:
        state = get_state(obs, stations[(stations.index(state[1]) + 1) % 4], True)
        inference_step = 0
    if not state[7] and state[0] not in stations and state[8] and previous_action == 5:
        state = get_state(obs, state[1], False)
        inference_step = 0

    action = None
    if state not in q_table:
        action = random.choice(range(6))
    else:
        action_probs = softmax(q_table[state])
        action = int(np.random.choice(len(action_probs), p=action_probs))

    previous_state = state
    previous_action = action

    return action