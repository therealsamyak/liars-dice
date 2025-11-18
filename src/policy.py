"""Optimal policy implementation for Player 1."""

from collections import Counter

from src.game_state import Bid, ObservableState
from src.probability import p_valid


class OptimalPolicy:
    """Optimal policy for Player 1."""

    def __init__(self, num_dice_per_player: int, liar_threshold: float = 0.1):
        self.num_dice_per_player = num_dice_per_player
        self.liar_threshold = liar_threshold

    def get_action(self, obs: ObservableState) -> Bid | None:
        """Returns Bid to make or None to call LIAR."""
        if obs.game_status.value != "Active":
            return None

        last_bid = obs.bid_history[-1] if obs.bid_history else None

        # If no previous bid, make a conservative first bid
        if last_bid is None:
            # Count our dice and make a bid based on what we have
            # Start with a conservative bid: one of our most common dice
            counts = Counter(obs.hand1)
            if counts:
                most_common_value, most_common_count = counts.most_common(1)[0]
                # Bid conservatively: what we have plus expected from opponent
                expected_opponent = (
                    self.num_dice_per_player // 6
                )  # Expected ~1/6 of opponent's dice
                amount = most_common_count + max(1, expected_opponent)
                return Bid(value=most_common_value, amount=amount)
            else:
                # Fallback: bid one 1
                return Bid(value=1, amount=1)

        # Calculate probability that the last bid is valid
        # D_last = last_bid, H1 = obs.hand1, N = self.num_dice_per_player
        prob_valid = p_valid(last_bid, obs.hand1, self.num_dice_per_player)

        # If probability is very low, call LIAR
        if prob_valid < self.liar_threshold:
            return None  # Signal to call LIAR

        # Otherwise, make a higher bid
        # Strategy: increase value if possible, otherwise increase amount
        if last_bid.value < 6:
            # Can increase value
            new_value = last_bid.value + 1
            # Make a conservative bid: what we have plus expected
            # c1 = COUNT(H1, v*)
            c1 = sum(1 for die in obs.hand1 if die == new_value)
            expected_opponent = self.num_dice_per_player // 6
            new_amount = c1 + max(1, expected_opponent)
            # Must be higher than last bid
            if new_value > last_bid.value or (
                new_value == last_bid.value and new_amount > last_bid.amount
            ):
                return Bid(value=new_value, amount=new_amount)

        # Increase amount on same value
        new_amount = last_bid.amount + 1
        return Bid(value=last_bid.value, amount=new_amount)
