
import util
from deepQLearningAgents import GhostDQAgent


class MultDQLAgent(GhostDQAgent):
    def __init__(self, index, epsilon=0.1, gamma=0.9, alpha=0.1, buffer_size=10000, batch_size=128, lr=0.001, numTraining=0, num_ghost=2, **args):
        features_num = num_ghost * 2 + 2  # 2 for Pacman's position, 2 for each ghost
        super().__init__(index, epsilon, gamma, alpha, buffer_size,
                         batch_size, lr, numTraining, features_num, **args)

    def extract_state_features(self, state):
        pacman_pos = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()

        # Include Pacman's position and all ghost positions as part of the state features
        features = [pacman_pos[0], pacman_pos[1]]
        for ghost_pos in ghost_positions:
            features.append(ghost_pos[0])
            features.append(ghost_pos[1])

        return features

    def update(self, state, action, next_state, reward, done):
        pacman_pos = next_state.getPacmanPosition()
        ghost_positions = next_state.getGhostPositions()

        # Calculate total distance to Pacman for all ghosts
        total_distance = sum([util.manhattanDistance(
            pacman_pos, ghost_pos) for ghost_pos in ghost_positions])

        # Check how many legal moves Pacman has left
        pacman_legal_moves = next_state.getLegalPacmanActions()
        if len(pacman_legal_moves) <= 2:  # If Pacman has 2 or fewer moves, reward the ghosts
            reward += 50

        # Reward all ghosts if they collectively trap Pacman (total distance = 0)
        if total_distance == 0:
            reward += 100

        # Encourage ambushing behavior: if ghosts are equidistant from Pacman, reward more
        distance_pairs = [util.manhattanDistance(
            g1, pacman_pos) for g1 in ghost_positions]
        if len(set(distance_pairs)) == 1:  # If multiple ghosts are equidistant from Pacman
            reward += 50  # Reward for ambushing Pacman

        reward -= total_distance  # Penalize for increasing total distance from Pacman

        self.memory.add((self.extract_state_features(
            state), self.action_to_index(action), reward, self.extract_state_features(next_state), done))

        if len(self.memory) > self.batch_size and self.timestep % 4 == 0:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
