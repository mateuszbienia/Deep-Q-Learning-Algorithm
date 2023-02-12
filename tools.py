import copy
from typing import List, Tuple
import numpy as np


def play_epiode_one_sided(env, agents: List, target, epsilon):
    obs, _ = copy.deepcopy(env.reset())
    terminal = truncated = False
    old_obs = copy.deepcopy(obs)
    episode_experience = []
    turn = 0
    while not (terminal or truncated):
        action = agents[0].get_action(obs, env.get_moves(), epsilon)
        obs, reward, terminal, truncated, _ = env.step(action)
        episode_experience.append([old_obs, action, reward, obs, False])
        turn += 1
        if (not terminal and not truncated):
            action_p2 = agents[1].get_action(obs, env.get_moves(), epsilon)
            observations_p2, reward, terminal, truncated, _ = env.step(
                action_p2)
            old_obs = copy.deepcopy(observations_p2)
            turn += 1
    episode_experience.append([obs, action, reward, None, True])
    for exp in episode_experience:
        agents[0].memory.add(*exp)
    agents[0].train(target)
    return turn, reward


def simulate_game(env, agents, visualize=False):
    done, terminated = False, False
    obs, _ = env.reset()
    turn = 0
    while not (done or terminated):
        action = agents[turn % 2].get_action(obs, env.get_moves())
        turn += 1
        obs, reward, done, terminated, _ = env.step(action)
        if visualize:
            env.render()
    if visualize:
        env.get_winner_info()
    if visualize and reward == -2:
        print("Player of type {} made illegal move".format(
            type(agents[(turn+1) % 2])))
    return reward, turn


def evaluate(env, agents, nrounds=100):
    rewards = []
    turns = []
    for i in range(nrounds):
        r, t = simulate_game(env, agents)
        rewards.append(r)
        turns.append(t)
    return rewards, turns


def test_winrate(env, agents: List, nrounds: int = 200, swap_sides: bool = True, info: bool = False) -> Tuple[float, float, float, float]:
    r, t = evaluate(env, agents, nrounds//2)
    p1 = np.sum([1 if d == 1 else 0 for d in r])/(nrounds//2)
    p2 = np.sum([1 if d == -1 else 0 for d in r])/(nrounds//2)
    illigal_p1 = np.sum([1 if d == -2 else 0 for d in r])
    illigal_p2 = np.sum([1 if d == -2 else 0 for d in r])
    if info:
        print(f"p1: {p1}    p2:  {p2}")
        print(f"illigal_p1: {illigal_p1}    illigal_p2:  {illigal_p2}")
        print("draw: {0:.2} ".format(1-p1 - illigal_p1 - p2 - illigal_p2))
    if swap_sides:
        agents[0], agents[1] = agents[1], agents[0]
        r, t = evaluate(env, agents, nrounds - nrounds//2)
        p2 += np.sum([1 if d == -1 else 0 for d in r])/(nrounds//2)
        p1 += np.sum([1 if d == 1 else 0 for d in r])/(nrounds//2)
        illigal_p2 += np.sum([1 if d == -2 else 0 for d in r])
        illigal_p1 += np.sum([1 if d == -2 else 0 for d in r])
        if info:
            print(f"p1: {p1/2}    p2:  {p2/2}")
            print(f"illigal_p1: {illigal_p1/2}    illigal_p2:  {illigal_p2/2}")
            print("draw: {0:.2} ".format(
                1-p1/2 - illigal_p1/2 - p2/2 - illigal_p2/2))
        return p1/2, p2/2, illigal_p1/2, illigal_p2/2
    else:
        return p1, p2, illigal_p1, illigal_p2
