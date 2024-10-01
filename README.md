

## Traffic Light Control with Reinforcement Learning

This project implements a traffic light control system using **Reinforcement Learning (RL)**, specifically the **Proximal Policy Optimization (PPO)** algorithm. The system is designed to optimize traffic flow by dynamically adjusting traffic light phases in a simulated urban environment using **SUMO** (Simulation of Urban MObility) and **traci** interface.

### Features
- **Traffic Environment Simulation**: Utilizes the SUMO simulator for modeling and simulating real-world traffic conditions.
- **Reinforcement Learning Agent**: Implements a PPO-based RL agent to optimize traffic light control decisions.
- **Dynamic Control**: The agent learns to adjust traffic light phases to minimize vehicle wait times and improve traffic flow efficiency.
- **Customizable Settings**: The simulation parameters (e.g., simulation time, traffic light durations, reward functions) are customizable for different use cases.
- **Modular Code Structure**: Clear separation of environment, main script, and PPO algorithm for easy modification and experimentation.

### Project Structure
- **`AdnaneEnv.py`**: Defines the traffic light environment and interaction with the SUMO simulator.
- **`main.py`**: Main script to initialize the environment, train the RL agent, and run simulation episodes.
- **`PPO.py`**: Implementation of the PPO algorithm, including actor-critic networks and training loop.

### Requirements
- **SUMO** (Simulation of Urban MObility)
- **traci** (SUMO's Python API)
- **numpy** (Numerical operations)
- **PyTorch** (for implementing PPO)

To install the necessary Python dependencies, run:
```bash
pip install traci numpy torch
```

### How to Use
1. **Set up SUMO**: Ensure SUMO is installed and set up on your system.
2. **Configure SUMO environment**: Modify the `sumocfg_file` in `AdnaneEnv.py` with your SUMO configuration file that defines the traffic network and demand.
3. **Run the simulation**: Execute `main.py` to start training the PPO agent and running traffic simulations:
   ```bash
   python main.py
   ```
4. **Optimize PPO**: The PPO algorithm requires optimization (e.g., hyperparameters like learning rate, policy clipping, and batch size) to achieve good results. Adjust these parameters in `PPO.py` for better performance based on your simulation environment.
5. **Customize Parameters**: Adjust environment parameters like `simulation_time`, `min_green`, and `reward_type` in `AdnaneEnv.py` to fit your specific simulation requirements.

### Future Work
- Fine-tuning the PPO algorithm for better traffic flow results.
- Adding more complex road networks and traffic conditions for training.
- Enhancing the reward function for more specific traffic goals (e.g., minimizing emissions, prioritizing emergency vehicles).
- Exploring other RL algorithms and comparing their performance with PPO.

### Contributions
Contributions, issues, and feature requests are welcome! Feel free to fork this repository and open a pull request.

