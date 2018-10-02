import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import gym


SOLVE_TAXI_MESSAGE = """Task : \n
1) The cab(YELLOW) should find the shortest path to BLUE(passenger) 
2) Perform a "pickup" action to board the passenger which turns the cab(GREEN)
3) Take the passenger to the PINK(drop location) using the shortest path
4) Perform a "dropoff" action
"""


def load_qtable(qtable_path):
    q_data = pd.read_csv(qtable_path, header=None)
    
    print(q_data.head())
    q_data = q_data.drop(columns=[0])
    q_data = q_data[q_data.index != 0]
    q_data.columns = list(range(0,6))
    return q_data.values


def select_best_action(qtable, state):
    return np.argmax(qtable[state])


def timestamp():
    return int(datetime.datetime.now().timestamp() * 100)


def clear_screen(delay=1):
    time.sleep(delay)
    os.system('clear')

    
def log_progress(env, reward=0, total_reward=0, delay=None, message=None):
    if type(message) is str:
        print(message)
    env.render()
    print('Reward:', reward)
    print('Cumulative reward', total_reward)
    clear_screen(delay)
    

def perf_message(attempt, perf):
    return '{}\nAttempt: {} | Average reward (until last episode): {:.2f}'.format(
        SOLVE_TAXI_MESSAGE, 
        attempt + 1, 
        perf
    )
    
def init_message(attempt, perf):
    return 'Initial State : {}'.format(perf_message(attempt=attempt, perf=perf))
    
def solve_taxi(env, qtable, attempt=None, perf=None):
    clear_screen(0)
    state = env.reset()
    done = False
    steps = 0
    log_progress(env, delay=0.5, message=init_message(attempt, perf))
    total_reward = 0
    while not done and steps <= 50:
        steps += 1
        action = select_best_action(qtable, state)
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        log_progress(env, reward=reward, total_reward=total_reward, delay=0.5,message=perf_message(attempt, perf))
        state = new_state
    clear_screen(0)
    return total_reward
        
    
def evaluate_solution(qtable_path, episodes=100):
    env = gym.make('Taxi-v2')
    qtable = load_qtable(qtable_path)
    score  = 0
    perf   = 0
    reward = 0
    reward_per_episode = np.zeros(episodes)
    for i in range(episodes):
        reward = solve_taxi(env, qtable, attempt=i, perf=perf)
        reward_per_episode[i] = reward
        score += reward
        perf = score/(i + 1)        
    print('Agent gets an average reward of {:.2f}'.format(perf))
    print(reward_per_episode)
    return perf

if __name__ == '__main__':
    print(sys.argv)
    qtable_path = sys.argv[1] if len(sys.argv) > 1 else None
    if qtable_path is not None:
        evaluate_solution(qtable_path)