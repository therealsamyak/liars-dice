"""RL-based policy wrapper for Liar's Dice."""

from src.game_state import Bid, ObservableState
from src.rl_agent import RLAgent


class RLPolicy:
    """Policy wrapper for RL agents that matches OptimalPolicy interface."""
    
    def __init__(self, rl_agent: RLAgent):
        """Initialize RL policy.
        
        Args:
            rl_agent: Underlying RL agent (Q-Learning, DQN, etc.)
        """
        self.rl_agent = rl_agent
        self.num_dice_per_player = rl_agent.num_dice_per_player
    
    def get_action(self, obs: ObservableState) -> Bid | None:
        """Get action from RL agent.
        
        Args:
            obs: Observable state
            
        Returns:
            Bid to make, or None to call LIAR
        """
        return self.rl_agent.get_action(obs)
    
    def set_training_mode(self, training: bool):
        """Set training vs evaluation mode.
        
        Args:
            training: True for training (exploration), False for evaluation
        """
        self.rl_agent.set_training_mode(training)
    
    def save(self, filepath: str):
        """Save RL agent.
        
        Args:
            filepath: Path to save file
        """
        self.rl_agent.save(filepath)
    
    def load(self, filepath: str):
        """Load RL agent.
        
        Args:
            filepath: Path to load file
        """
        self.rl_agent.load(filepath)
