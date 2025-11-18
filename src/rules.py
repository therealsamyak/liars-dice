"""Game rules and validation for Liar's Dice."""

from src.game_state import Bid, GameStatus, State


def count_dice(hand: list[int], value: int) -> int:
    """Count the number of dice in hand showing the given value."""
    return sum(1 for die in hand if die == value)


def count_total_dice(hand1: list[int], hand2: list[int], value: int) -> int:
    """Count total dice across both hands showing the given value."""
    return count_dice(hand1, value) + count_dice(hand2, value)


def is_legal_bid(new_bid: Bid, last_bid: Bid | None) -> bool:
    """
    Check if a bid is legal according to the rules.

    Rules:
    - First bid (last_bid is None): any valid bid allowed
    - Must increase dice value OR increase quantity (or both)
    - Cannot decrease both value and quantity
    """
    if last_bid is None:
        # First bid - any valid bid is allowed
        return True

    # Must increase value OR increase quantity (or both)
    if new_bid.value > last_bid.value:
        return True  # Increased value

    if new_bid.value == last_bid.value and new_bid.amount > last_bid.amount:
        return True  # Same value, increased quantity

    return False  # Invalid bid


def check_win_condition(
    state: State,
    caller: int,  # 1 for Player 1, 2 for Player 2
    last_bid: Bid,
) -> int | None:
    """
    Determine the winner after a LIAR call.

    Returns:
        - 1 if Player 1 wins
        - 2 if Player 2 wins
        - None if game is still active (shouldn't happen)
    """
    if state.game_status != GameStatus.GAME_OVER:
        return None

    total_count = count_total_dice(state.hand1, state.hand2, last_bid.value)

    # If the bid is valid (total_count >= amount), the bidder wins
    # If the bid is invalid (total_count < amount), the caller wins
    if total_count >= last_bid.amount:
        # Bid was valid - bidder wins
        # If Player 1 called LIAR, Player 2 (the bidder) wins
        # If Player 2 called LIAR, Player 1 (the bidder) wins
        return 2 if caller == 1 else 1
    else:
        # Bid was invalid - caller wins
        return caller


def is_bid_valid(state: State, bid: Bid) -> bool:
    """Check if a bid is actually valid given the true game state."""
    total_count = count_total_dice(state.hand1, state.hand2, bid.value)
    return total_count >= bid.amount
