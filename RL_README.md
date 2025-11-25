# Reinforcement Learning for Liar's Dice

This document describes how to use the reinforcement learning capabilities in this Liar's Dice implementation.

## Overview

The project now supports **Q-Learning** as a reinforcement learning algorithm alongside the existing Dynamic Programming (DP) solver. The RL agent learns through self-play and can be compared against:
- Truth-telling policy
- Optimal DP policy
- Other RL agents

## Training an RL Agent

### Basic Training

Train a Q-Learning agent with default parameters:

```bash
uv run train_rl.py
```

This will:
- Train for 10,000 episodes (configurable in `parameters.py`)
- Play against truth-telling opponent
- Save checkpoints every 1,000 episodes to `rl_models/`
- Generate learning curve visualizations

### Advanced Training Options

```bash
# Train for more episodes
uv run train_rl.py --episodes 50000

# Train against optimal DP policy
uv run train_rl.py --opponent optimal

# Resume from checkpoint
uv run train_rl.py --load-checkpoint rl_models/q_agent_episode_5000.json

# Custom save interval
uv run train_rl.py --save-interval 2000

# Custom output directory
uv run train_rl.py --output-dir my_models
```

### Training Parameters

Modify `parameters.py` to adjust RL hyperparameters:

```python
RL_LEARNING_RATE = 0.1          # Learning rate (alpha)
RL_DISCOUNT_FACTOR = 0.95       # Discount factor (gamma)
RL_EPSILON = 0.1                # Initial exploration rate
RL_EPSILON_DECAY = 0.995        # Exploration decay per episode
RL_EPSILON_MIN = 0.01           # Minimum exploration rate
RL_TRAINING_EPISODES = 10000    # Default training episodes
```

## Evaluating RL Agents

### Run Simulations

Compare trained RL agents against other policies:

```bash
# RL agent (Player 1) vs Truth-telling (Player 2)
uv run main.py --player1-policy rl --player2-policy truth_telling

# RL agent vs Optimal DP policy
uv run main.py --player1-policy rl --player2-policy optimal

# RL vs RL (self-play evaluation)
uv run main.py --player1-policy rl --player2-policy rl

# Specify custom RL model
uv run main.py --player1-policy rl --rl-model-p1 rl_models/q_agent_episode_5000.json

# Run more simulations
uv run main.py --player1-policy rl --num-simulations 1000

# Verbose mode (show game details)
uv run main.py --player1-policy rl --verbose
```

### Example Output

```
=== Liar's Dice Simulation ===

Player 1: RL Agent (loaded from rl_models/q_agent_final.json)
Player 2: Truth-Telling Policy

Running 200 simulations...
  - 100 vs truth-telling policy (50 P1 start, 50 P2 start)
  - 100 vs optimal policy (50 P1 start, 50 P2 start)
============================================================
Completed 10/200 simulations...
...
```

## Architecture

### Components

1. **`src/rl_agent.py`** - Abstract base class for RL agents
2. **`src/q_learning_agent.py`** - Tabular Q-Learning implementation
3. **`src/rl_policy.py`** - Policy wrapper for RL agents
4. **`src/rl_utils.py`** - Utility functions (state representation, action encoding)
5. **`train_rl.py`** - Training script with trajectory tracking
6. **`main.py`** - Updated to support RL agent evaluation

### State Representation

RL agents use a feature-based state representation:
- Hand counts: `[count of 1s, count of 2s, ..., count of 6s]`
- Last bid: `(value, amount)` or `(0, 0)` if no bid
- Bid history length

### Actions

- **Make a bid**: `(value, amount)` tuple
- **Call LIAR**: `(0, 0)` special encoding

### Learning Algorithm

Q-Learning update rule:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

- **Exploration**: ε-greedy policy
- **Reward**: +1 for win, -1 for loss
- **Discount factor**: Same as DP solver (0.95)

## Comparison: RL vs DP

| Aspect | Q-Learning (RL) | Dynamic Programming (DP) |
|--------|----------------|--------------------------|
| **Approach** | Learn through experience | Enumerate all states |
| **Training** | Required (self-play) | No training needed |
| **State Space** | Scalable with function approximation | Limited by enumeration |
| **Optimality** | Converges with sufficient training | Provably optimal |
| **Computation** | O(episodes × game length) | O(states × actions) |
| **Flexibility** | Adapts to opponents | Fixed policy |

## Tips for Better Performance

1. **Training Duration**: Train for at least 10,000 episodes for decent performance
2. **Opponent Variety**: Train against both truth-telling and optimal policies
3. **Exploration**: Start with higher ε (0.3-0.5) for better exploration
4. **Learning Rate**: Lower learning rate (0.01-0.05) for more stable learning
5. **State Space**: Current implementation works well for 2 dice per player

## Troubleshooting

**Issue**: RL agent performs poorly
- **Solution**: Train for more episodes or adjust hyperparameters

**Issue**: Training is slow
- **Solution**: Reduce number of episodes or use a smaller state space

**Issue**: Model file not found
- **Solution**: Train a model first using `python train_rl.py`

**Issue**: Q-table grows too large
- **Solution**: Reduce `MAX_BID_AMOUNT` or implement function approximation (DQN)

## Future Enhancements

- Deep Q-Network (DQN) for larger state spaces
- Policy gradient methods (REINFORCE, PPO)
- Multi-agent RL (self-play with evolving opponents)
- Transfer learning between different game configurations
