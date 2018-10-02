from collections import deque
import sys
import math
import numpy as np
import pandas as pd
import datetime


def save_rewards_csv(average_reward_per_100_episodes, best_average_reward_per_100_episodes):
    episodes = np.arange(1, len(average_reward_per_100_episodes) + 1)
    reward_per_episode_df = pd.DataFrame({
        'average_reward_per_100_episodes': average_reward_per_100_episodes,
        'best_average_reward_per_100_episodes': best_average_reward_per_100_episodes,
        'episodes': episodes
    })
    reward_per_episode_df.to_csv('rewards_plot_data.csv')


def interact(env, agent, num_episodes=50000, window=100):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    average_reward_per_100_episodes = []
    best_average_reward_per_100_episodes = []
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent selects an action
            action = agent.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            agent.update_epsilon()
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break

        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            print('episode average reward {}'.format(avg_reward))
            average_reward_per_100_episodes.append(avg_reward)
            best_average_reward_per_100_episodes.append(best_avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {} || eps {} ".format(i_episode, num_episodes, best_avg_reward, agent.epsilon), end="")
        sys.stdout.flush()
        
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            agent.q_table.save(best_avg_reward)
            save_rewards_csv(average_reward_per_100_episodes, best_average_reward_per_100_episodes)
            break
        if i_episode == num_episodes: 
            agent.q_table.save(best_avg_reward)
            save_rewards_csv(average_reward_per_100_episodes, best_average_reward_per_100_episodes)
            print('\n')
    return avg_rewards, best_avg_reward