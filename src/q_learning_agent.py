"""Q-Learning agent for Liar's Dice."""

import json
import random
from collections import defaultdict

from src.game_state import Bid, ObservableState
from src.rl_agent import RLAgent
from src.rl_utils import (
    action_to_tuple,
    get_all_possible_actions,
    state_to_hash,
    tuple_to_action,
)


class QLearningAgent(RLAgent):
    """Tabular Q-Learning agent using epsilon-greedy exploration."""
    
    def __init__(
        self,
        num_dice_per_player: int,
        player_id: int = 1,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """Initialize Q-Learning agent.
        
        Args:
            num_dice_per_player: Number of dice per player
            player_id: ID of this player (1 or 2)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        super().__init__(num_dice_per_player, player_id)
        
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: dict[state_hash, dict[action_tuple, float]]
        self.q_table: dict[str, dict[tuple[int, int], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        
        # Track statistics
        self.total_updates = 0
        self.episodes_trained = 0
        
    def get_action(self, obs: ObservableState) -> Bid | None:
        """Select action using epsilon-greedy policy.
        
        Args:
            obs: Current observable state
            
        Returns:
            Bid to make, or None to call LIAR
        """
        state_hash = state_to_hash(obs, self.num_dice_per_player)
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        
        # Get all legal actions
        max_amount = self.num_dice_per_player * 2  # 2 players
        legal_actions = get_all_possible_actions(last_bid, max_amount=max_amount)
        
        # Epsilon-greedy exploration
        if self.training_mode and random.random() < self.epsilon:
            # Explore: random action
            action_tuple = random.choice(legal_actions)
        else:
            # Exploit: best known action
            action_tuple = self._get_best_action(state_hash, legal_actions)
        
        return tuple_to_action(action_tuple)
    
    def _get_best_action(
        self, 
        state_hash: str, 
        legal_actions: list[tuple[int, int]]
    ) -> tuple[int, int]:
        """Get best action based on Q-values.
        
        Args:
            state_hash: Hash of current state
            legal_actions: List of legal action tuples
            
        Returns:
            Best action tuple
        """
        # Get Q-values for all legal actions
        q_values = [
            self.q_table[state_hash][action] for action in legal_actions
        ]
        
        # Find max Q-value
        max_q = max(q_values)
        
        # Get all actions with max Q-value (handle ties randomly)
        best_actions = [
            action for action, q in zip(legal_actions, q_values) if q == max_q
        ]
        
        return random.choice(best_actions)
    
    def learn(
        self,
        state: ObservableState,
        action: Bid | None,
        reward: float,
        next_state: ObservableState | None,
        done: bool,
    ):
        """Update Q-table using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        
        Args:
            state: State before action
            action: Action taken
            reward: Immediate reward
            next_state: State after action (None if terminal)
            done: Whether episode is done
        """
        state_hash = state_to_hash(state, self.num_dice_per_player)
        action_tuple = action_to_tuple(action)
        
        # Current Q-value
        current_q = self.q_table[state_hash][action_tuple]
        
        # Compute target Q-value
        if done or next_state is None:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: add discounted max future Q-value
            next_state_hash = state_to_hash(next_state, self.num_dice_per_player)
            last_bid = next_state.bid_history[-1] if next_state.bid_history else None
            max_amount = self.num_dice_per_player * 2
            next_legal_actions = get_all_possible_actions(
                last_bid, max_amount=max_amount
            )
            
            # Max Q-value over next legal actions
            if next_legal_actions:
                max_next_q = max(
                    self.q_table[next_state_hash][a] for a in next_legal_actions
                )
            else:
                max_next_q = 0.0
            
            target_q = reward + self.gamma * max_next_q
        
        # Q-Learning update
        self.q_table[state_hash][action_tuple] += self.alpha * (
            target_q - current_q
        )
        
        self.total_updates += 1
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_episode(self):
        """Reset for new episode and decay epsilon."""
        self.episodes_trained += 1
        self.decay_epsilon()
    
    def save(self, filepath: str):
        """Save Q-table and parameters to JSON file.
        
        Args:
            filepath: Path to save file
        """
        # Convert defaultdict to regular dict for JSON serialization
        q_table_serializable = {}
        for state_hash, actions in self.q_table.items():
            q_table_serializable[state_hash] = {
                f"{a[0]},{a[1]}": q for a, q in actions.items()
            }
        
        data = {
            "num_dice_per_player": self.num_dice_per_player,
            "player_id": self.player_id,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "total_updates": self.total_updates,
            "episodes_trained": self.episodes_trained,
            "q_table": q_table_serializable,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(
            f"Q-Learning agent saved to {filepath} "
            f"({len(self.q_table)} states, {self.total_updates} updates)"
        )
    
    def load(self, filepath: str):
        """Load Q-table and parameters from JSON file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.num_dice_per_player = data["num_dice_per_player"]
        self.player_id = data["player_id"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.epsilon_min = data["epsilon_min"]
        self.total_updates = data["total_updates"]
        self.episodes_trained = data["episodes_trained"]
        
        # Reconstruct Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_hash, actions in data["q_table"].items():
            for action_str, q_value in actions.items():
                value, amount = map(int, action_str.split(","))
                self.q_table[state_hash][(value, amount)] = q_value
        
        print(
            f"Q-Learning agent loaded from {filepath} "
            f"({len(self.q_table)} states, {self.total_updates} updates)"
        )
    
    def get_stats(self) -> dict:
        """Get agent statistics.
        
        Returns:
            Dictionary with agent stats
        """
        return {
            "episodes_trained": self.episodes_trained,
            "total_updates": self.total_updates,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "total_state_actions": sum(
                len(actions) for actions in self.q_table.values()
            ),
        }
