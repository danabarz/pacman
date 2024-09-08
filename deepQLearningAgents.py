import torch
import torch.nn as nn
import torch.optim as optim
import random
from learningAgents import ReinforcementAgent
import util
import numpy as np
from collections import deque

ACTION_NUMBER = 4
FEATURE_SIZE = 5
ACTION_MAP = {
    'North': 0,
    'South': 1,
    'East': 2,
    'West': 3
}


# Define the Deep Q-Network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.memory[idx] for idx in indices])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# Deep Q-Learning Agent Class
class GhostDQAgent(ReinforcementAgent):
    def __init__(self, index, epsilon=0.1, gamma=0.9, alpha=0.1, buffer_size=10000, batch_size=128, lr=0.001, numTraining=0, features_num=FEATURE_SIZE, **args):
        super().__init__(numTraining=numTraining,
                         epsilon=epsilon, alpha=alpha, gamma=gamma, **args)

        self.index = index
        self.state_size = features_num
        self.action_size = ACTION_NUMBER
        self.lr = lr
        self.batch_size = batch_size
        self.timestep = 0

        self.memory = ReplayBuffer(buffer_size)
        self.qnetwork_local = DQNetwork(self.state_size, self.action_size)
        self.qnetwork_target = DQNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.lr)
        self.tau = 0.001  # For soft update of target network

    def getAction(self, state):
        """Choose action based on epsilon-greedy strategy"""
        legal_actions = state.getLegalActions(self.index)

        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)

        # Exploitation: predict Q-values and select the best legal action
        state_tensor = torch.FloatTensor(
            self.extract_state_features(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)

        # Only consider legal actions
        q_values = q_values.detach().cpu().numpy().flatten()

        # Find the best legal action by comparing Q-values for each legal action
        best_legal_action = None
        max_q_value = float('-inf')
        for action in legal_actions:
            # Map the action to its index in the Q-value list
            action_index = self.action_to_index(action)
            if q_values[action_index] > max_q_value:
                max_q_value = q_values[action_index]
                best_legal_action = action

        self.doAction(state, best_legal_action)
        return best_legal_action

    def action_to_index(self, action):
        """
        Map actions to their corresponding index in the Q-value list.
        Example: if you have 4 possible actions (NORTH, SOUTH, EAST, WEST),
        map each one to an index.
        """
        return ACTION_MAP.get(action, None)

    def extract_state_features(self, state):
        """This function should extract the relevant features from the state for input to the neural network.
        For example, this could include the positions of the ghosts, Pacman, food, etc."""
        pacman_pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPosition(self.index)
        scared_timer = state.getGhostState(self.index).scaredTimer
        features = [pacman_pos[0], pacman_pos[1],
                    ghost_pos[0], ghost_pos[1], scared_timer]
        return features

    def update(self, state, action, next_state, reward, done):
        """Store experience and perform learning"""
        # next_pacman_pos = next_state.getPacmanPosition()
        # next_ghost_pos = next_state.getGhostPosition(self.index)
        # distance = util.manhattanDistance(next_pacman_pos, next_ghost_pos)
        # reward -= distance

        if state.isWin() or next_state.isWin():
            reward -= 1

        if state.isLose() or next_state.isLose():
            reward += 1

        self.memory.add((self.extract_state_features(
            state), self.action_to_index(action), reward, self.extract_state_features(next_state), done))

        if len(self.memory) > self.batch_size and self.timestep % 4 == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def is_trapping_pacman(self, next_state):
        """
        Example heuristic: Check if Pacman has limited movement options.
        For instance, Pacman is trapped when he has fewer than 2 legal moves.
        """
        legal_pacman_moves = next_state.getLegalPacmanActions()
        # Trapping when Pacman has only 1-2 moves left
        return len(legal_pacman_moves) <= 2

    def learn(self, experiences):
        """Update network weights"""
        states, actions, rewards, next_states, dones = experiences

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get max predicted Q values for next states from target model
        next_q_values = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        # Compute target Q values
        target_q_values = rewards + \
            (self.discount * next_q_values * (1 - dones))

        # Get expected Q values from local model
        expected_q_values = self.qnetwork_local(states).gather(1, actions)

        # Compute loss and perform backward pass
        loss = nn.MSELoss()(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard update of target network
        self.timestep += 1
        if self.timestep % 100 == 0:
            self.qnetwork_target.load_state_dict(
                self.qnetwork_local.state_dict())

    def final(self, state):
        """This method can be used to save the model or perform any final steps after training"""
        ReinforcementAgent.final(self, state)
        if self.episodesSoFar % 100 == 0:
            print("Training finished. Saving model...")
            torch.save(self.qnetwork_local.state_dict(),
                       f"dqn_model_ghost_{self.index}.pth")
