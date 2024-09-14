import torch
import torch.nn as nn
import torch.optim as optim
from game import Directions
from learningAgents import ReinforcementAgent
from deepQLearningAgents import ReplayBuffer
import numpy as np
import util
import os

ACTION_MAP = {
    0: 'North',
    1: 'South',
    2: 'East',
    3: 'West'
}


class CentralizedDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CentralizedDQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


class CentralizedDQLAgent(ReinforcementAgent):
    def __init__(self, state_size=11, action_size=16, lr=0.001, gamma=0.99, epsilon=0.1,
                  buffer_size=1000, batch_size=64, numTraining=0, checkpoint=None, dqn_model=None, **args):
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
        self.update_every = 4  # Frequency of learning updates
        self.target_update_freq = 1000  # Frequency of updating target network
        self.epsilon = epsilon  # Exploration rate
        self.alpha = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.decay_rate = 0.995  # Decay rate for epsilon and learning rate

        if dqn_model:
            self.dqn_model = dqn_model
            self.load_model(self.dqn_model)
            print(f"Model loaded successfully: {self.dqn_model}")
        else:
            print("No model provided. Initializing new model.")

        if checkpoint:
            self.load_model(checkpoint)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(0.05, self.epsilon * self.decay_rate)

    def adjust_alpha(self):
        """Decay learning rate after each episode if needed."""
        self.alpha = max(0.01, self.alpha * self.decay_rate)

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

    def is_trapping_pacman(self, next_state):
        """
        Example heuristic: Check if Pacman has limited movement options.
        For instance, Pacman is trapped when he has fewer than 2 legal moves.
        """
        legal_pacman_moves = next_state.getLegalPacmanActions()
        return len(legal_pacman_moves) <= 2

    def split_joint_action(self, joint_action):
        # Ensure joint_action is an integer
        joint_action = int(joint_action)
        ghost1_action = joint_action // 4
        ghost2_action = joint_action % 4
        return ghost1_action, ghost2_action

    def update(self, state, action, next_state, reward, done):
        """Store experience and perform learning for two ghosts"""
        state_features = self.extract_state_features(state)
        next_state_features = self.extract_state_features(next_state)
        pacman_pos = state.getPacmanPosition()
        ghost_positions = [state.getGhostPosition(
            i + 1) for i in range(len(state.getGhostStates()))]

        # Reward for reducing distance to Pacman
        dist_ghost1 = util.manhattanDistance(pacman_pos, ghost_positions[0])
        dist_ghost2 = util.manhattanDistance(pacman_pos, ghost_positions[1])
        new_dist_ghost1 = util.manhattanDistance(
            next_state.getPacmanPosition(), next_state.getGhostPosition(1))
        new_dist_ghost2 = util.manhattanDistance(
            next_state.getPacmanPosition(), next_state.getGhostPosition(2))

        # Reward getting closer to Pacman
        if new_dist_ghost1 < dist_ghost1:
            reward += 5
        if new_dist_ghost2 < dist_ghost2:
            reward += 5

        # Penalize if ghosts get further from Pacman
        if new_dist_ghost1 > dist_ghost1:
            reward -= 5
        if new_dist_ghost2 > dist_ghost2:
            reward -= 5

        # Reward for ambush positioning (ghosts coming from opposite sides)
        if dist_ghost1 < 2 or dist_ghost2 < 2:
            if (ghost_positions[0][0] < pacman_pos[0] < ghost_positions[1][0] or ghost_positions[0][0] > pacman_pos[0] > ghost_positions[1][0]) or \
                    (ghost_positions[0][1] < pacman_pos[1] < ghost_positions[1][1] or ghost_positions[0][1] > pacman_pos[1] > ghost_positions[1][1]):
                reward += 50
            else:
                reward += 20

        if self.is_trapping_pacman(next_state):
            reward += 50

        # Add experience and learn from it
        self.memory.add((state_features, self.action_to_index(
            action), reward, next_state_features, done))
        self.timestep += 1

        # Learning and updating
        if len(self.memory) > self.batch_size and self.timestep % self.update_every == 0:
            experiences = self.sample_experiences()
            self.learn(experiences)

        if self.timestep % self.target_update_freq == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target)

        return reward

    def sample_experiences(self):
        """Sample experiences from memory"""
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
        return (torch.FloatTensor(states), torch.LongTensor(actions).unsqueeze(1), torch.FloatTensor(rewards).unsqueeze(1), torch.FloatTensor(next_states), torch.FloatTensor(dones).unsqueeze(1))

    def learn(self, experiences):
        """Update network weights"""
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        expected_q_values = self.qnetwork_local(states).gather(1, actions)
        loss = nn.MSELoss()(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model):
        """ Soft update of target network: θ_target = τ*θ_local + (1 - τ)*θ_target """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def extract_state_features(self, state):
        """Extract relevant features from the game state for input to the neural network."""
        pacman_pos = state.getPacmanPosition()
        pacman_direction = state.getPacmanState().getDirection()
        ghost_states = state.getGhostStates()

        features = [pacman_pos[0], pacman_pos[1]]
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            features.append(ghost_pos[0])
            features.append(ghost_pos[1])

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

        if len(ghost_states) > 1:
            ghost1_pos = ghost_states[0].getPosition()
            ghost2_pos = ghost_states[1].getPosition()
            features.append(util.manhattanDistance(ghost1_pos, ghost2_pos))

        return features

    def final(self, state):
        """Finalize the episode and save the model after training."""
        ReinforcementAgent.final(self, state)
        if self.episodesSoFar % 100 == 0 and self.episodesSoFar != 0 and self.episodesSoFar <=self.numTraining:
            print("Training finished. Saving model...")
            self.save_model(f"dqn_model_centralized_1.pth")

    def save_model(self, filepath):
        """Save the model and replay buffer to disk."""
        if os.path.exists(filepath):
            print("Deleting previous save at", filepath)
            os.remove(filepath)
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': self.memory
        }, filepath)
        print("Model saved to", filepath)

    def load_model(self, filepath):
        """Load the model and replay buffer from disk."""
        print("Loading model from", filepath)
        checkpoint = torch.load(filepath, weights_only=False)
        self.qnetwork_local.load_state_dict(
            checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(
            checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.memory = checkpoint['replay_buffer']

    def setTrainingMode(self, mode):
        self.training = mode
