"""Base class for reinforcement learning agents."""

from abc import ABC, abstractmethod
from src.game_state import Bid, ObservableState


class RLAgent(ABC):
    """Abstract base class for RL agents in Liar's Dice."""
    
    def __init__(self, num_dice_per_player: int, player_id: int = 1):
        """Initialize RL agent.
        
        Args:
            num_dice_per_player: Number of dice each player has
            player_id: ID of this player (1 or 2)
        """
        self.num_dice_per_player = num_dice_per_player
        self.player_id = player_id
        self.training_mode = True  # Default to training mode
        
    @abstractmethod
    def get_action(self, obs: ObservableState) -> Bid | None:
        """Select action given observable state.
        
        Args:
            obs: Current observable state
            
        Returns:
            Bid to make, or None to call LIAR
        """
        pass
    
    @abstractmethod
    def learn(
        self, 
        state: ObservableState, 
        action: Bid | None, 
        reward: float, 
        next_state: ObservableState | None,
        done: bool
    ):
        """Update agent based on experience.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action (None if terminal)
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save agent to file.
        
        Args:
            filepath: Path to save file
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent from file.
        
        Args:
            filepath: Path to load file
        """
        pass
    
    def set_training_mode(self, training: bool):
        """Set training mode (exploration) vs evaluation mode (exploitation).
        
        Args:
            training: True for training mode, False for evaluation
        """
        self.training_mode = training
    
    def reset_episode(self):
        """Reset agent state for new episode (optional, override if needed)."""
        pass
