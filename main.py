import numpy as np
import torch
import csv
import os
from AdnaneEnv import AdnaneEnv  # Assuming AdnaneEnv is saved in a file named AdnaneEnv.py
from PPO import PPOAgent  # Assuming the PPOAgent class from the repository is saved in PPO.py

def save_to_csv(data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["episode", "step", "agent", "accumulating_waiting_time", "reward"])
        writer.writerows(data)

def main():
    # Initialize the environment
    env = AdnaneEnv(sumocfg_file='v1.sumocfg', simulation_time=20000, min_green=5, yellow_time=2, gui=True, reward_type='compititf')
    num_agents = env.get_num_traffic_lights()
    traffic_light_ids = env.get_traffic_lights_ids()
    observations, rewards, _ = env.reset()

    # Define PPO parameters
    input_dims = {tl_id: observations[tl_id].shape for tl_id in traffic_light_ids}
    n_actions = {tl_id: env.get_num_actions(tl_id) for tl_id in traffic_light_ids}
    lr = 0.0003
    gamma = 0.99
    eps_clip = 0.2
    batch_size = 64
    n_epochs = 10
    done = True
    episode_data = []

    # Initialize PPO agents for each traffic light
    # Initialize PPO agents for each traffic light
    agents = {tl_id: PPOAgent(n_actions[tl_id], input_dims[tl_id], gamma, lr, 0.95, eps_clip, batch_size, n_epochs) for tl_id in traffic_light_ids}

    # Load models if they exist
    for tl_id in traffic_light_ids:
        agents[tl_id].load_models(agent_name=f'_1_{tl_id}')
    # Training parameters
    max_episodes = 1000
    jsp = 1000

    for episode in range(max_episodes):
        # Reset environment
        if done:
            print("\n", done)
            observations, rewards, done = env.reset()


        for step in range(jsp):
            actions = {}
            probs = {}
            vals = {}
            for tl_id in traffic_light_ids:
                action, prob, val = agents[tl_id].choose_action(observations[tl_id])
                actions[tl_id] = action
                probs[tl_id] = prob
                vals[tl_id] = val

            # Perform step
            new_observations, rewards, done = env.step(actions)

            for tl_id in traffic_light_ids:
                # Store transition in agent's memory
                agents[tl_id].remember(observations[tl_id], actions[tl_id], probs[tl_id], vals[tl_id], rewards[tl_id], done)

                # Collect data for CSV
                accumulating_waiting_time = env.get_waiting_time(tl_id)
                episode_data.append([episode, step, tl_id, accumulating_waiting_time, rewards[tl_id]])

            # Update observations
            observations = new_observations
            for tl_id in traffic_light_ids:
                agents[tl_id].learn()
            if done:
                break
            

        # Update PPO agents after each episode
        

        # Save models and data after each episode
        if episode % 100 == 0:
            print(f'Episode {episode} completed')
            for tl_id in traffic_light_ids:
                #agents[tl_id].save_models(f'_{episode}_{tl_id}')

            # Save episode data to CSV
            #for tl_id in traffic_light_ids:
                filename = f'agent_{tl_id}_{episode}_data.csv'
                #save_to_csv(episode_data, filename)
            episode_data = []


    print("Training completed")

if __name__ == '__main__':
    main()
