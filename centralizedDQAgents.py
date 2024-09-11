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
        # Increased size for better learning
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class CentralizedDQLAgent(ReinforcementAgent):
    def __init__(self, state_size=20, action_size=16, lr=0.0001, gamma=0.99, epsilon=0.1, buffer_size=10000, batch_size=64, numTraining=0, **args):
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

    def update(self, state, action, next_state, reward, done):
        """Store experience and perform learning for multiple ghosts"""
        state_features = self.extract_state_features(state)
        next_state_features = self.extract_state_features(next_state)

        pacman_pos = next_state.getPacmanPosition()
        ghost_positions = [next_state.getGhostPosition(
            i + 1) for i in range(len(next_state.getGhostStates()))]
        reward = 0

        if len(ghost_positions) == 2:
            ghost1_pos = ghost_positions[0]
            ghost2_pos = ghost_positions[1]
            dist_ghost1 = util.manhattanDistance(pacman_pos, ghost1_pos)
            dist_ghost2 = util.manhattanDistance(pacman_pos, ghost2_pos)

            if dist_ghost1 < 3 and dist_ghost2 < 3:
                if (ghost1_pos[0] < pacman_pos[0] < ghost2_pos[0] or ghost1_pos[0] > pacman_pos[0] > ghost2_pos[0]) or \
                        (ghost1_pos[1] < pacman_pos[1] < ghost2_pos[1] or ghost1_pos[1] > pacman_pos[1] > ghost2_pos[1]):
                    reward += 50
                else:
                    reward += 10

        for ghost_pos in ghost_positions:
            dist_ghost = util.manhattanDistance(pacman_pos, ghost_pos)
            reward += (10 - dist_ghost)

        if state.getNumFood() > next_state.getNumFood():
            reward -= 15

        if len(state.getCapsules()) > len(next_state.getCapsules()):
            reward -= 25

        ghost_to_pacman_distances = [util.manhattanDistance(
            pacman_pos, ghost_pos) for ghost_pos in ghost_positions]
        if min(ghost_to_pacman_distances) > 5:
            reward -= 5

        self.memory.add((state_features, self.action_to_index(action), reward,
                        next_state_features, done))

        if len(self.memory) > self.batch_size and self.timestep % 2 == 0:
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

    # def extract_state_features(self, state):
    #     """Extract relevant features from the game state for input to the neural network."""
    #     pacman_pos = state.getPacmanPosition()
    #     pacman_direction = state.getPacmanState().getDirection()
    #     ghost_states = state.getGhostStates()
    #     food = state.getFood()
    #     # capsules = state.getCapsules()
    #     walls = state.getWalls()

    #     features = [pacman_pos[0], pacman_pos[1]]
    #     for ghost_state in ghost_states:
    #         ghost_pos = ghost_state.getPosition()
    #         features.append(ghost_pos[0])
    #         features.append(ghost_pos[1])

    #     food_distances = [util.manhattanDistance(
    #         pacman_pos, food_pos) for food_pos in food.asList()]
    #     features.append(min(food_distances) if food_distances else 0)
    #     # features.append(len(capsules))

    #     direction_map = {
    #         Directions.NORTH: 0,
    #         Directions.SOUTH: 1,
    #         Directions.EAST: 2,
    #         Directions.WEST: 3
    #     }

    #     direction_encoded = [0, 0, 0, 0]
    #     if pacman_direction in direction_map:
    #         direction_encoded[direction_map[pacman_direction]] = 1
    #     features.extend(direction_encoded)

    #     # Add ghost-to-ghost distances for better collaboration
    #     if len(ghost_states) > 1:
    #         ghost1_pos = ghost_states[0].getPosition()
    #         ghost2_pos = ghost_states[1].getPosition()
    #         features.append(util.manhattanDistance(ghost1_pos, ghost2_pos))

    #     for dx in [-1, 0, 1]:
    #         for dy in [-1, 0, 1]:
    #             x = int(pacman_pos[0] + dx)
    #             y = int(pacman_pos[1] + dy)
    #             if 0 <= x < walls.width and 0 <= y < walls.height:
    #                 features.append(1 if walls[x][y] else 0)
    #             else:
    #                 features.append(1)

    #     return features

    def final(self, state):
        """Finalize the episode and save the model after training."""
        ReinforcementAgent.final(self, state)
        if self.episodesSoFar % 100 == 0:
            print("Training finished. Saving model...")
            torch.save(self.qnetwork_local.state_dict(),
                       f"dqn_model_centralized.pth")

    def extract_state_features(self, state):
        # Pacman's position and direction
        pacman_pos = state.getPacmanPosition()
        pacman_direction = state.getPacmanState().getDirection()

        # Ghosts' positions and relative distances to Pacman
        ghost_states = state.getGhostStates()
        ghost_positions = [ghost_state.getPosition()
                           for ghost_state in ghost_states]

        features = [pacman_pos[0], pacman_pos[1]]  # Pacman's x, y coordinates

        # Encode Pacman's direction as a one-hot vector (N, S, E, W)
        direction_map = {
            Directions.NORTH: [1, 0, 0, 0],
            Directions.SOUTH: [0, 1, 0, 0],
            Directions.EAST:  [0, 0, 1, 0],
            Directions.WEST:  [0, 0, 0, 1]
        }
        direction_encoded = direction_map.get(pacman_direction, [0, 0, 0, 0])
        # Add Pacman's direction to features
        features.extend(direction_encoded)

        # Ghosts' positions relative to Pacman
        for ghost_pos in ghost_positions:
            ghost_pacman_dist_x = ghost_pos[0] - pacman_pos[0]
            ghost_pacman_dist_y = ghost_pos[1] - pacman_pos[1]
            features.append(ghost_pacman_dist_x)
            features.append(ghost_pacman_dist_y)

            # Ghost-to-Ghost distances
            if len(ghost_positions) > 1:
                ghost1_pos = ghost_positions[0]
                ghost2_pos = ghost_positions[1]
                ghost_dist = util.manhattanDistance(ghost1_pos, ghost2_pos)
                features.append(ghost_dist)  # Add ghost-to-ghost distance

            # Food distances (minimum distance to the nearest food)
            food_positions = state.getFood().asList()
            food_distances = [util.manhattanDistance(
                pacman_pos, food_pos) for food_pos in food_positions]
            if food_distances:
                # Minimum distance to food
                features.append(min(food_distances))
            else:
                features.append(0)  # If no food is left, set to 0

            # Capsule distances (minimum distance to the nearest capsule)
            capsule_positions = state.getCapsules()
            capsule_distances = [util.manhattanDistance(
                pacman_pos, capsule_pos) for capsule_pos in capsule_positions]
            if capsule_distances:
                # Minimum distance to capsule
                features.append(min(capsule_distances))
            else:
                features.append(0)  # If no capsules are left, set to 0

            # Wall information (for the local area around Pacman)
            walls = state.getWalls()
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x = int(pacman_pos[0] + dx)
                    y = int(pacman_pos[1] + dy)
                    if 0 <= x < walls.width and 0 <= y < walls.height:
                        # 1 if wall is present, else 0
                        features.append(1 if walls[x][y] else 0)
                    else:
                        features.append(1)  # Boundary walls treated as solid

            return features
