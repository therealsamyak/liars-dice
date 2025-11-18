"""Probability calculations for optimal policy."""

from math import comb

from src.game_state import Bid


def p_valid(D_last: Bid, H1: list[int], N: int) -> float:
    """
    Calculate P_valid: probability that a bid is valid given our hand.

    P_valid(D_last | H1) = P(COUNT(H1 ∪ H2, v*) >= a* | H1)

    Where:
    - D_last = (v*, a*) - value and amount
    - c1 = COUNT(H1, v*) - number of dice in our hand with value v*
    - c2 = COUNT(H2, v*) - number of dice in opponent's hand with value v*
    - k = a* - c1 - additional dice needed from opponent for bid to be valid
    - P_valid = P(c2 >= k) where c2 ~ Binomial(N, p=1/6)

    P_valid = P(c2 >= k) = sum_{i=k}^{N} C(N, i) * (1/6)^i * (5/6)^(N-i)

    Special cases:
    - If k > N, then P_valid = 0 (impossible bid)
    - If k <= 0, then P_valid = 1 (our hand satisfies bid)

    Args:
        D_last: The bid to evaluate (D_last = (v*, a*))
        H1: Our hand (H1)
        N: Number of dice each player has (N)

    Returns:
        Probability that the bid is valid (0.0 to 1.0)
    """
    v_star = D_last.value  # DiceValue
    a_star = D_last.amount  # Amount
    c1 = sum(1 for die in H1 if die == v_star)  # COUNT(H1, v*)
    k = a_star - c1  # additional dice needed from opponent

    # If k > N, impossible bid
    if k > N:
        return 0.0

    # If k <= 0, our hand already satisfies the bid
    if k <= 0:
        return 1.0

    # Calculate P(c2 >= k) = sum_{i=k}^{N} C(N, i) * (1/6)^i * (5/6)^(N-i)
    p = 1.0 / 6.0
    q = 5.0 / 6.0
    total_prob = 0.0

    for i in range(k, N + 1):
        binomial_coeff = comb(N, i)
        prob = binomial_coeff * (p**i) * (q ** (N - i))
        total_prob += prob

    return total_prob
