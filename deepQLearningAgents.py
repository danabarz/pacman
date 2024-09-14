import torch
import torch.nn as nn
import torch.optim as optim
import random
from learningAgents import ReinforcementAgent
import util
import numpy as np
from collections import deque
import os

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
    def __init__(self, index, epsilon=0.1, gamma=0.9, alpha=0.1, buffer_size=10000, batch_size=128, lr=0.001,
                 numTraining=0, features_num=5, action_size=4, checkpoint=None, dqn_model = None, **args):
        super().__init__(numTraining=numTraining,
                         epsilon=epsilon, alpha=alpha, gamma=gamma, **args)

        self.index = index
        self.state_size = features_num
        self.action_size = action_size
        self.lr = lr
        self.batch_size = batch_size
        self.timestep = 0
        self.memory = ReplayBuffer(buffer_size)
        self.qnetwork_local = DQNetwork(self.state_size, self.action_size)
        self.qnetwork_target = DQNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(
        self.qnetwork_local.parameters(), lr=self.lr)
        if dqn_model:
            self.dqn_model = dqn_model
            self.load_model(self.dqn_model)
            print(f"Model loaded successfully: {self.dqn_model}")
        else:
            print("No model provided. Initializing new model.")
        if checkpoint:
            self.load_model(checkpoint)


    def getAction(self, state, agentIndex=1):
        """Choose action based on epsilon-greedy strategy"""
        legal_actions = state.getLegalActions(self.index)

        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)

        state_tensor = torch.FloatTensor(
            self.extract_state_features(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork_local(state_tensor)
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
        self.memory.add((self.extract_state_features(
            state), self.action_to_index(action), reward, self.extract_state_features(next_state), done))

        if len(self.memory) > self.batch_size and self.timestep % 4 == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        return reward

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
                       f"dqn_model_ghost_.pth")
            
    def save_model(self, filepath):
        """Save the model and replay buffer to disk."""
        if os.path.exists(filepath):
            print("Deleting previous save at", filepath)
            os.remove(filepath)
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': list(self.memory.memory)  # Convert deque to list for serialization
        }, filepath)
        print("Model saved to", filepath)

    def load_model(self, filepath):
        """Load the model and replay buffer from disk."""
        print("Loading model from", filepath)
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        print("Checkpoint keys:", checkpoint.keys())  # Print all keys in the checkpoint

        # Check for the presence of expected keys
        if 'qnetwork_local_state_dict' in checkpoint:
            self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
            self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory.memory = deque(checkpoint['replay_buffer'], maxlen=self.memory.memory.maxlen)  # Restore deque
        else:
            # Fallback: assume the checkpoint contains only the state_dict of the model
            self.qnetwork_local.load_state_dict(checkpoint)
            self.qnetwork_target.load_state_dict(checkpoint)
            print("Loaded model weights only, no optimizer or replay buffer state.")

        print("Model loaded successfully from", filepath)

