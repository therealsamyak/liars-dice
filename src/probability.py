"""Probability calculations for optimal policy."""

from math import comb

from src.game_state import Bid


def p_valid(bid: Bid, hand1: list[int], num_dice_per_player: int) -> float:
    """
    Calculate P_valid: probability that a bid is valid given our hand.

    P_valid(D_last | H1) = P(COUNT(H1 ∪ H2, v*) >= a* | H1)

    Where:
    - c1 = COUNT(H1, v*)
    - k = a* - c1 (additional dice needed from opponent)
    - P_valid = P(c2 >= k) where c2 ~ Binomial(N, p=1/6)

    Args:
        bid: The bid to evaluate (D_last)
        hand1: Our hand (H1)
        num_dice_per_player: Number of dice each player has (N)

    Returns:
        Probability that the bid is valid (0.0 to 1.0)
    """
    c1 = sum(1 for die in hand1 if die == bid.value)
    k = bid.amount - c1

    # If k > N, impossible bid
    if k > num_dice_per_player:
        return 0.0

    # If k <= 0, our hand already satisfies the bid
    if k <= 0:
        return 1.0

    # Calculate P(c2 >= k) = sum_{i=k}^{N} C(N, i) * (1/6)^i * (5/6)^(N-i)
    p = 1.0 / 6.0
    q = 5.0 / 6.0
    total_prob = 0.0

    for i in range(k, num_dice_per_player + 1):
        binomial_coeff = comb(num_dice_per_player, i)
        prob = binomial_coeff * (p**i) * (q ** (num_dice_per_player - i))
        total_prob += prob

    return total_prob
