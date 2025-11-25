"""Utilities for reinforcement learning agents."""

from src.game_state import Bid, ObservableState


def state_to_features(obs: ObservableState, num_dice_per_player: int) -> tuple:
    """Convert observable state to feature tuple for RL agents.
    
    Features:
    - Hand counts: [count of 1s, count of 2s, ..., count of 6s]
    - Last bid: (value, amount) or (0, 0) if no bid
    - Bid history length
    
    Returns:
        Immutable tuple of features suitable for hashing (Q-table keys)
    """
    # Count dice in hand
    hand_counts = [0] * 6
    for die in obs.hand1:
        hand_counts[die - 1] += 1
    
    # Last bid info
    if obs.bid_history:
        last_bid = obs.bid_history[-1]
        last_bid_value = last_bid.value
        last_bid_amount = last_bid.amount
    else:
        last_bid_value = 0
        last_bid_amount = 0
    
    # Bid history length (indicates game progress)
    bid_history_len = len(obs.bid_history)
    
    # Return immutable tuple
    return tuple(hand_counts + [last_bid_value, last_bid_amount, bid_history_len])


def state_to_hash(obs: ObservableState, num_dice_per_player: int) -> str:
    """Create unique hash string for state (for Q-table indexing).
    
    Args:
        obs: Observable state
        num_dice_per_player: Number of dice per player
        
    Returns:
        String hash of state
    """
    features = state_to_features(obs, num_dice_per_player)
    return "_".join(map(str, features))


def action_to_tuple(action: Bid | None) -> tuple[int, int]:
    """Convert action to tuple representation.
    
    Args:
        action: Bid or None (call LIAR)
        
    Returns:
        (value, amount) tuple, or (0, 0) for LIAR
    """
    if action is None:
        return (0, 0)  # Special encoding for LIAR
    return (action.value, action.amount)


def tuple_to_action(action_tuple: tuple[int, int]) -> Bid | None:
    """Convert tuple representation back to action.
    
    Args:
        action_tuple: (value, amount) tuple
        
    Returns:
        Bid or None (call LIAR)
    """
    if action_tuple == (0, 0):
        return None  # LIAR
    return Bid(value=action_tuple[0], amount=action_tuple[1])


def compute_training_reward(winner: int, player_id: int = 1) -> float:
    """Compute reward for training.
    
    Args:
        winner: Player who won (1 or 2)
        player_id: ID of player being trained (default 1)
        
    Returns:
        Reward value: +1 for win, -1 for loss
    """
    if winner == player_id:
        return 1.0
    else:
        return -1.0


def get_all_possible_actions(
    last_bid: Bid | None, 
    max_value: int = 6,
    max_amount: int = 4  # Default for 2 dice per player
) -> list[tuple[int, int]]:
    """Get all possible actions from current state.
    
    Args:
        last_bid: Last bid made, or None if first turn
        max_value: Maximum die value (default 6)
        max_amount: Maximum bid amount (default 4 for 2 players × 2 dice)
        
    Returns:
        List of action tuples: [(value, amount), ...] including (0,0) for LIAR
    """
    actions = []
    
    # Option 1: Call LIAR (if there's a bid to challenge)
    if last_bid is not None:
        actions.append((0, 0))
    
    # Option 2: Make a bid
    if last_bid is None:
        # First bid: any valid bid
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                actions.append((value, amount))
    else:
        # Increase value (any amount allowed)
        for value in range(last_bid.value + 1, max_value + 1):
            for amount in range(1, max_amount + 1):
                actions.append((value, amount))
        
        # Same value, increase amount
        for amount in range(last_bid.amount + 1, max_amount + 1):
            actions.append((last_bid.value, amount))
    
    return actions
