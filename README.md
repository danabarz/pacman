
# Pacman AI Project

This project implements AI agents for controlling ghosts in the Pacman game. The agents utilize various algorithms such as Q-Learning and Deep Q-Learning to enhance their behavior.

## How to Run the Project

To run the Pacman game with AI-controlled ghosts, you can use the following command:

```bash
python pacman.py [options]
```

### Example Usage:

1. **Start an interactive game:**
   ```bash
   python pacman.py
   ```

2. **Start a game with a specific layout and zoom level:**
   ```bash
   python pacman.py --layout smallClassic --zoom 2
   ```

### Available Options:

- `-n`, `--numGames`: Number of games to play (default: 1)
  - Example: `--numGames 10`
  
- `-l`, `--layout`: Specifies the layout file for the map (default: mediumClassic)
  - Example: `--layout smallClassic`
  
- `-p`, `--pacman`: Specifies the Pacman agent type (default: GreedyAgent)
  - Example: `--pacman GreedyAgent`
  
- `-g`, `--ghosts`: Specifies the ghost agent type (default: RandomGhost)
  - Example: `--ghosts CentralizedDQLAgent`
  
- `-t`, `--textGraphics`: Runs the game in text-only mode.
  - Example: `--textGraphics`
  
- `-q`, `--quietTextGraphics`: Runs the game in quiet text mode with minimal output.
  - Example: `--quietTextGraphics`
  
- `-z`, `--zoom`: Sets the zoom level for the game window (default: 1.0)
  - Example: `--zoom 2.0`
  
- `-f`, `--fixRandomSeed`: Fixes the random seed for reproducible results.
  - Example: `--fixRandomSeed`
  
- `-a`, `--agentArgs`: Arguments for the agent (e.g., epsilon, alpha).
  - Example: `--agentArgs epsilon=0.1,alpha=0.5`

- `-k`, `--numghosts`: The maximum number of ghosts to use (default: 4).
  - Example: `--numghosts 1`

- `-r`, `--recordActions`: Writes game histories to a file (named by the time they were played)
  - Example: `--recordActions`

- `--replay`: A recorded game file (pickle) to replay.
  - Example: `--replay filepath`

- `-x`, `--numTraining`: How many episodes are training (suppresses output) (default: 0).
  - Example: `--numTraining 100`

- `--timeout`: Maximum length of time an agent can spend computing in a single game (default: 30).
  - Example: `--timeout 60`

- `--dqn_model`: Path to the DQN model file to load.
  - Example: `--dqn_model filepath`

- `--q_table_name`: Path to the Q-table file to load/save.
  - Example: `--q_table_name filepath`

## Example with Agents

### Pacman Agents
- GreedyAgent:
  ```bash
  python pacman.py -p GreedyAgent
  ```

### Ghost Agents

- **BFSGhost - BaseLine**: 
Specify `-k` as the number of ghosts value (e.g., 1), -l to choose layout (smallGrid or mediumClassic is possible).
  ```bash
  python pacman.py -g BFSGhost -k 1 -l smallGrid
  ```

- **AStarGhost**: 
Specify `-k` as the number of ghosts value (e.g., 1), -l to choose layout (smallGrid or mediumClassic is possible).
  ```bash
  python pacman.py -g AStarGhost -k 1 -l smallGrid
  ```

- **MinMaxGhost**: 
Specify `-k` as the number of ghosts value (e.g., 1), -l to choose layout (smallGrid or mediumClassic is possible).
  ```bash
  python pacman.py -g MinMaxGhost
  ```

- **GhostQAgent**: 
  Include the file to load the Q-table with `--q_table_name`, specify `-n` for number of games, with `-k` as the number of ghosts value (e.g., 1), with layout smallGrid.
  ```bash
  python pacman.py -g GhostQAgent --q_table_name qtable.pkl -x 50 -k 1 -l smallGrid
  ```

- **GhostDQAgent**: 
  Include the file to load the DQN model with `--dqn_model`, specify `-n` for number of games,  with `-k` as the number of ghosts value (e.g., 2),
  -l to choose layout (smallGrid or mediumClassic is possible).
  ```bash
  python pacman.py -g GhostDQAgent --dqn_model dqn_model_ghost_.pth -n 10 -k 1 -l smallGrid
  ```

- **CentralizedDQLAgent**: 
  Include the file to load the DQN model with `--dqn_model`, specify `-n` for number of games,  with `-k` as the number of ghosts value (e.g., 2), , with layout mediumClassic.
  ```bash
  python pacman.py -g CentralizedDQLAgent --dqn_model centralized_dqn_model_1.pth -n 10 -k 2 -l mediumClassic
  ```

## Additional Information

- This project requires several Python libraries. Ensure you have all dependencies installed by running:

```bash
pip install -r requirements.txt
```

### Requirements
The `requirements.txt` file contains the necessary packages for this project.
