"""Game rules and validation for Liar's Dice."""

from src.game_state import Bid, GameStatus, State


def count_dice(H: list[int], v: int) -> int:
    """COUNT(H, v): returns the number of dice in hand H showing value v."""
    return sum(1 for die in H if die == v)


def count_total_dice(H1: list[int], H2: list[int], v: int) -> int:
    """
    COUNT(H1 ∪ H2, v): total dice across both hands showing value v.

    COUNT(H1 ∪ H2, v) = COUNT(H1, v) + COUNT(H2, v)
    """
    return count_dice(H1, v) + count_dice(H2, v)


def is_legal_bid(new_bid: Bid, D_last: Bid | None) -> bool:
    """
    LEGAL(D): Check if a bid is legal according to the rules.

    Standard Liar's Dice bidding rules:
    - Each subsequent bid must increase either quantity OR face value
    - You can increase quantity while keeping same face value
    - You can increase face value with ANY quantity (even 1)
    - You CANNOT decrease face value
    - You CANNOT decrease quantity when keeping same face value

    Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current (any amount allowed when increasing value)
        True  if DiceValue_new = DiceValue_current and Amount_new > Amount_current
        False otherwise
    }
    """
    if D_last is None:
        # D = ∅: first bid - any valid bid is allowed
        return True

    # DiceValue_new > DiceValue_current
    # When increasing value, any amount is allowed (standard Liar's Dice rule)
    if new_bid.value > D_last.value:
        return True

    # DiceValue_new = DiceValue_current and Amount_new > Amount_current
    # When keeping same value, must increase amount
    if new_bid.value == D_last.value and new_bid.amount > D_last.amount:
        return True

    return False


def check_win_condition(
    state: State,
    caller: int,  # 1 for Player 1, 2 for Player 2
    D_last: Bid,
) -> int | None:
    """
    WIN(s, caller): Determine the winner after a LIAR call.

    Matches MATH.md win conditions:
    - If caller = 1 and COUNT < Amount: P1 wins (caller was correct)
    - If caller = 1 and COUNT >= Amount: P2 wins (bidder was correct)
    - If caller = 2 and COUNT >= Amount: P1 wins (bidder was correct)
    - If caller = 2 and COUNT < Amount: P2 wins (caller was correct)

    WIN(s, caller) = {
        P1  if caller = 1, G_o = GameOver, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        P2  if caller = 1, G_o = GameOver, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P1  if caller = 2, G_o = GameOver, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P2  if caller = 2, G_o = GameOver, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        None if G_o = Active
    }

    Returns:
        - 1 if Player 1 wins (P1)
        - 2 if Player 2 wins (P2)
        - None if game is still active (shouldn't happen)
    """
    if state.game_status != GameStatus.GAME_OVER:
        return None

    # COUNT(H1 ∪ H2, D_last.Value)
    total_count = count_total_dice(state.hand1, state.hand2, D_last.value)

    # If COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount, bidder wins
    # If COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount, caller wins
    if total_count >= D_last.amount:
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
