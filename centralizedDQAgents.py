import torch
import torch.nn as nn
import torch.optim as optim
from game import Directions
from learningAgents import ReinforcementAgent
from deepQLearningAgents import ReplayBuffer
import numpy as np
import util


ACTION_MAP = {
    0: 'North',
    1: 'South',
    2: 'East',
    3: 'West'
}


class CentralizedDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CentralizedDQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# lr=0.0001


class CentralizedDQLAgent(ReinforcementAgent):
    def __init__(self, state_size=21, action_size=16, lr=0.0001, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=64, numTraining=100, **args):
        super().__init__(numTraining=numTraining,
                         epsilon=epsilon, gamma=gamma, **args)
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        self.qnetwork_local = CentralizedDQNetwork(state_size, action_size)
        self.qnetwork_target = CentralizedDQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.tau = 0.001
        self.timestep = 0

    def add_to_memory(self, experience):
        """Automatically manages buffer size with deque"""
        self.memory.append(experience)

    def getAction(self, state, agentIndex=1):
        """ Epsilon-greedy action selection for centralized control of multiple ghosts """
        state_features = self.extract_state_features(state)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            joint_action = np.random.choice(self.action_size)
        else:
            q_values = self.qnetwork_local(state_tensor)
            joint_action = np.argmax(q_values.detach().numpy())

        ghost1_action, ghost2_action = self.split_joint_action(joint_action)
        legal_actions = state.getLegalActions(agentIndex)

        if agentIndex == 1:
            action = ACTION_MAP[ghost1_action]
        elif agentIndex == 2:
            action = ACTION_MAP[ghost2_action]

        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        self.doAction(state, action)
        return action

    def split_joint_action(self, joint_action):
        """ Split joint action into individual actions for each ghost """
        ghost1_action = joint_action // 4
        ghost2_action = joint_action % 4
        return ghost1_action, ghost2_action

    # def update(self, state, action, next_state, reward, done):
    #     """Store experience and perform learning"""
    #     # Modify reward to encourage ambushing
    #     pacman_pos = next_state.getPacmanPosition()
    #     ghost_positions = next_state.getGhostPositions()

    #     # Compute distances from Pacman to each ghost
    #     distances = [util.manhattanDistance(
    #         pacman_pos, ghost_pos) for ghost_pos in ghost_positions]

    #     # Reward when both ghosts are close to Pacman (ambushing)
    #     if min(distances) < 2 and max(distances) < 4:
    #         reward += 10  # Reward for effective ambushing

    #     # Penalize if ghosts move far away from Pacman
    #     if max(distances) > 5:
    #         reward -= 5

    #     # Penalize for Pacman eating food
    #     if state.getNumFood() > next_state.getNumFood():
    #         reward -= 10

    #     # Call the regular update process
    #     state_features = self.extract_state_features(state)
    #     next_state_features = self.extract_state_features(next_state)
    #     self.memory.add((state_features, self.action_to_index(action), reward,
    #                     next_state_features, done))

    #     if len(self.memory) > self.batch_size and self.timestep % 4 == 0:
    #         experiences = self.sample_experiences()
    #         self.learn(experiences)

    #     self.timestep += 1

    def update(self, state, action, next_state, reward, done):
        """Store experience and perform learning for multiple ghosts"""
        state_features = self.extract_state_features(state)
        next_state_features = self.extract_state_features(next_state)
        pacman_pos = next_state.getPacmanPosition()

        num_ghosts = next_state.getNumAgents() - 1  # Pacman is agent 0, so subtract 1
        print(f"Pacman Position: {pacman_pos}")
        for ghost_index in range(1, num_ghosts + 1):
            ghost_pos = next_state.getGhostPosition(ghost_index)
            print(f"Ghost{i} Position: {ghost_pos}")
            dist_ghost = util.manhattanDistance(pacman_pos, ghost_pos)
            print(f"Distance Ghost{i}-Pacman: {dist_ghost}")
            reward += (10 - dist_ghost)

        if next_state.isLose():
            reward += 100

        if state.getNumFood() > next_state.getNumFood():
            reward -= 10

        # # Encourage the ghosts to trap Pacman by reducing his legal actions
        # pacman_legal_actions = len(next_state.getLegalPacmanActions())
        # reward += (4 - pacman_legal_actions)  # Reward is higher when Pacman has fewer options

        self.memory.add((state_features, self.action_to_index(action), reward,
                        next_state_features, done))

        if len(self.memory) > self.batch_size and self.timestep % 4 == 0:
            experiences = self.sample_experiences()
            self.learn(experiences)

        self.timestep += 1

    def sample_experiences(self):
        """Sample experiences from memory"""
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences

        return (torch.FloatTensor(states),
                torch.LongTensor(actions).unsqueeze(1),
                torch.FloatTensor(rewards).unsqueeze(1),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1))

    def learn(self, experiences):
        """Update network weights"""
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + \
            (self.discount * next_q_values * (1 - dones))
        expected_q_values = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """ Soft update of target network: θ_target = τ*θ_local + (1 - τ)*θ_target """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def extract_state_features(self, state):
        """Extract relevant features from the game state for input to the neural network."""
        pacman_pos = state.getPacmanPosition()
        pacman_direction = state.getPacmanState().getDirection()  # Get Pacman's direction
        ghost_positions = state.getGhostPositions()
        walls = state.getWalls()
        features = [pacman_pos[0], pacman_pos[1]]

        for ghost_pos in ghost_positions:
            features.append(ghost_pos[0])
            features.append(ghost_pos[1])

        for ghost_pos in ghost_positions:
            features.append(util.manhattanDistance(pacman_pos, ghost_pos))

        direction_map = {
            Directions.NORTH: 0,
            Directions.SOUTH: 1,
            Directions.EAST: 2,
            Directions.WEST: 3
        }

        direction_encoded = [0, 0, 0, 0]
        if pacman_direction in direction_map:
            direction_encoded[direction_map[pacman_direction]] = 1
        features.extend(direction_encoded)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x = int(pacman_pos[0] + dx)
                y = int(pacman_pos[1] + dy)
                if 0 <= x < walls.width and 0 <= y < walls.height:
                    features.append(1 if walls[x][y] else 0)
                else:
                    features.append(1)

        return features

    # def extract_state_features(self, state): # add this after i run with the new update
    #     """Extract relevant features from the game state for input to the neural network."""
    #     pacman_pos = state.getPacmanPosition()
    #     pacman_direction = state.getPacmanState().getDirection()
    #     ghost_states = state.getGhostStates()
    #     food = state.getFood()
    #     capsules = state.getCapsules()
    #     walls = state.getWalls()

    #     # Initialize feature list with Pacman's position
    #     features = [pacman_pos[0], pacman_pos[1]]

    #     # Add ghost positions and scared timers
    #     for ghost_state in ghost_states:
    #         ghost_pos = ghost_state.getPosition()
    #         features.append(ghost_pos[0])
    #         features.append(ghost_pos[1])
    #         features.append(ghost_state.scaredTimer)  # Scared timer feature

    #     # Add distance to closest food as a feature
    #     food_distances = [util.manhattanDistance(pacman_pos, food_pos) for food_pos in food.asList()]
    #     if food_distances:
    #         features.append(min(food_distances))
    #     else:
    #         features.append(0)  # No food left

    #     # Add number of capsules remaining
    #     features.append(len(capsules))

    #     # Encode Pacman's direction
    #     direction_map = {
    #         Directions.NORTH: 0,
    #         Directions.SOUTH: 1,
    #         Directions.EAST: 2,
    #         Directions.WEST: 3
    #     }
    #     direction_encoded = [0, 0, 0, 0]  # One-hot encoding for direction
    #     if pacman_direction in direction_map:
    #         direction_encoded[direction_map[pacman_direction]] = 1
    #     features.extend(direction_encoded)

    #     # Add wall information around Pacman (3x3 grid around Pacman)
    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             x = int(pacman_pos[0] + dx)
    #             y = int(pacman_pos[1] + dy)
    #             if 0 <= x < walls.width and 0 <= y < walls.height:
    #                 features.append(1 if walls[x][y] else 0)  # 1 for wall, 0 for no wall
    #             else:
    #                 features.append(1)

    #     # Print debugging information (optional)
    #     print(f"Pacman Position: {pacman_pos}, Direction: {pacman_direction}")
    #     for i, ghost_state in enumerate(ghost_states, 1):
    #         ghost_pos = ghost_state.getPosition()
    #         dist_ghost = util.manhattanDistance(pacman_pos, ghost_pos)
    #         print(f"Ghost{i} Position: {ghost_pos}, Distance to Pacman: {dist_ghost}, Scared Timer: {ghost_state.scaredTimer}")

    #     return features

    def final(self, state):
        """Finalize the episode and save the model after training."""
        ReinforcementAgent.final(self, state)
        if self.episodesSoFar % 100 == 0:
            print("Training finished. Saving model...")
            torch.save(self.qnetwork_local.state_dict(),
                       f"dqn_model_centralized.pth")
