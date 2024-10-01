import traci
import numpy as np

class AdnaneEnv:
    def __init__(self, sumocfg_file, simulation_time, min_green, yellow_time, gui, reward_type):
        self.sumocfg_file = sumocfg_file
        self.simulation_time = simulation_time
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.gui = gui
        self.reward_type = reward_type

        self.is_train_fonction = False
        traci.start(['sumo', '-c', self.sumocfg_file])
        self.traffic_light_ids = traci.trafficlight.getIDList()
        self.action_trafic = {tl_id: 0 for tl_id in self.traffic_light_ids}
        self.old_reward = {tl_id: 0 for tl_id in self.traffic_light_ids}
        self.is_yellow = {tl_id: False for tl_id in self.traffic_light_ids}
        self.time_since_last_change_yellow = {tl_id: 0 for tl_id in self.traffic_light_ids}
        self.time_since_last_change_green = {tl_id: 0 for tl_id in self.traffic_light_ids}
        traci.close()

    def get_num_traffic_lights(self):
        return len(self.traffic_light_ids)

    def get_traffic_lights_ids(self):
        return self.traffic_light_ids

    def get_num_actions(self, tl_id):
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        logic = logics[0]
        phases_list = logic.phases
        return int(len(phases_list) / 2)

    def step(self, actions):
        for tl_id, action in actions.items():
            self.change_traffic_light_state(tl_id, action)
            self.is_train_fonction = False

        for tl_id in self.traffic_light_ids:
            if self.is_yellow[tl_id]:
                self.time_since_last_change_yellow[tl_id] += 1
            else:
                self.time_since_last_change_green[tl_id] += 1
            traci.simulationStep()

        rewards = {tl_id: self.calculate_reward(tl_id) for tl_id in self.traffic_light_ids}
        observations = {tl_id: self.get_observation(tl_id) for tl_id in self.traffic_light_ids}
        done = self.is_done()

        return observations, rewards, done

    def calculate_reward(self, tl_id):
        wait_time = self.get_waiting_time(tl_id)
        if self.reward_type == 'Colaboratif':
            alpha = 0.75
            beta = 1.0
            total_congestion = self.get_total_congestion()
            R_individual = -wait_time
            R_collective = -total_congestion
            R_total = alpha * R_individual + beta * R_collective
        else:
            R_total = -wait_time

        reward = self.old_reward[tl_id] + R_total
        self.old_reward[tl_id] = R_total
        return reward

    def get_waiting_time(self, traffic_light_id):
        cumulative_waiting_time = 0
        stopped_vehicles = 0
        lane_ids = traci.trafficlight.getControlledLanes(traffic_light_id)
        for lane_id in lane_ids:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicle_ids:
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                cumulative_waiting_time += waiting_time
                if speed < 0.1:
                    stopped_vehicles += 1
        cumulative_waiting = (cumulative_waiting_time + stopped_vehicles)
        return cumulative_waiting

    def get_total_congestion(self):
        total_congestion = 0
        for tl_id in self.traffic_light_ids:
            lane_ids = traci.trafficlight.getControlledLanes(tl_id)
            for lane_id in lane_ids:
                lane_congestion = traci.lane.getLastStepOccupancy(lane_id)
                total_congestion += lane_congestion
        return total_congestion

    def change_traffic_light_state(self, tl_id, action):
        if self.is_yellow[tl_id]:
            if self.time_since_last_change_yellow[tl_id] >= self.yellow_time:
                action = self.action_trafic[tl_id]
                traci.trafficlight.setPhase(tl_id, action * 2)
                self.is_yellow[tl_id] = False
                self.time_since_last_change_yellow[tl_id] = 0
                self.time_since_last_change_green[tl_id] = 0
            else:
                self.time_since_last_change_yellow[tl_id] += 1
        else:
            if traci.trafficlight.getPhase(tl_id) != action * 2 and self.time_since_last_change_green[tl_id] >= self.min_green:
                yellow_phase = traci.trafficlight.getPhase(tl_id) + 1
                traci.trafficlight.setPhase(tl_id, yellow_phase)
                self.action_trafic[tl_id] = action
                self.is_yellow[tl_id] = True
                self.time_since_last_change_green[tl_id] = 0
            else:
                self.time_since_last_change_green[tl_id] += 1

    def get_observation(self, traffic_light_id):
        lane_speeds = []
        lane_vehicle_counts = []
        lane_ids = traci.trafficlight.getControlledLanes(traffic_light_id)
        for lane_id in lane_ids:
            lane_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            lane_vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_speeds.append(lane_speed)
            lane_vehicle_counts.append(lane_vehicle_count)

        observation = np.array(lane_speeds + lane_vehicle_counts)
        current_phase = traci.trafficlight.getPhase(traffic_light_id)
        time_in_phase = self.get_time_in_phase(traffic_light_id)
        if time_in_phase >= self.min_green:
            ready_to_change_traffic_light = True
        else:
            ready_to_change_traffic_light = False

        observation = np.append(observation, current_phase)
        observation = np.append(observation, time_in_phase)
        observation = np.append(observation, ready_to_change_traffic_light)
        return observation

    def get_time_in_phase(self, traffic_light_id):
        return traci.trafficlight.getSpentDuration(traffic_light_id)

    def is_done(self):
        return traci.simulation.getTime() >= self.simulation_time

    def reset(self):
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            print('TraCI is not running')
            pass

        sumo_command = ['sumo-gui' if self.gui else 'sumo', '-c', self.sumocfg_file]
        traci.start(sumo_command)
        self.traffic_light_ids = traci.trafficlight.getIDList()
        self.last_measure = {tl_id: 0 for tl_id in self.traffic_light_ids}
        self.is_yellow = {tl_id: False for tl_id in self.traffic_light_ids}
        self.time_since_last_change_yellow = {tl_id: 0 for tl_id in self.traffic_light_ids}
        self.time_since_last_change_green = {tl_id: 0 for tl_id in self.traffic_light_ids}
        actions = {tl_id: 1 for tl_id in self.traffic_light_ids}
        
        observations, rewards , done = self.step(actions)  # Only return the observations here
        
        return observations , rewards , done

