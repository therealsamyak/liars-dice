"""Opponent policy implementations."""

from src.game_state import Bid
from src.rules import count_dice, is_legal_bid


class TruthTellingPolicy:
    """Opponent policy: lowest truthful legal bid, otherwise lowest legal bid."""

    def __init__(self, num_dice_per_player: int):
        self.num_dice_per_player = num_dice_per_player

    def get_action(self, hand2: list[int], bid_history: list[Bid]) -> Bid | None:
        """Returns Bid to make or None to call LIAR.

        Strategy:
        1. If last bid is impossible given our hand, call LIAR
        2. Find lowest legal bid that is truthful (can be verified as true)
        3. If no truthful bid exists, find lowest legal bid (even if lying)
        """
        last_bid = bid_history[-1] if bid_history else None

        # If no previous bid, make lowest truthful first bid
        if last_bid is None:
            return self._lowest_truthful_first_bid(hand2)

        # Check if last bid is impossible (definitely invalid)
        # Bid is impossible if: our dice + max possible from opponent < bid amount
        c2_last = count_dice(hand2, last_bid.value)
        max_possible_from_opponent = self.num_dice_per_player
        if c2_last + max_possible_from_opponent < last_bid.amount:
            return None  # Call LIAR

        # Find lowest truthful legal bid
        truthful_bid = self._find_lowest_truthful_bid(hand2, last_bid)
        if truthful_bid is not None:
            return truthful_bid

        # No truthful bid found - find lowest legal bid (even if lying)
        return self._find_lowest_legal_bid(last_bid)

    def _lowest_truthful_first_bid(self, hand2: list[int]) -> Bid:
        """Find lowest truthful first bid."""
        # Try bids in order: (1,1), (1,2), ..., (1,N), (2,1), (2,2), ...
        for value in range(1, 7):
            for amount in range(1, self.num_dice_per_player * 2 + 1):
                bid = Bid(value=value, amount=amount)
                # Check if this bid is truthful given our hand
                c2 = count_dice(hand2, value)
                # Conservative estimate: we have c2, opponent could have up to N
                # So total could be c2 + N
                if c2 + self.num_dice_per_player >= amount:
                    return bid
        # Fallback (shouldn't happen)
        return Bid(value=1, amount=1)

    def _find_lowest_truthful_bid(self, hand2: list[int], last_bid: Bid) -> Bid | None:
        """Find lowest legal bid that is truthful.

        A bid is truthful if: COUNT(H2, value) + N >= amount
        (we have c2 dice, opponent could have up to N dice of this value)

        When increasing value, amount must be >= current amount (per MATH.md).
        """
        # Try bids in priority order: increase value first, then increase amount
        # Priority: (value+1, amount), (value+1, amount+1), ..., (6, amount), (6, amount+1), ...
        #          then (value, amount+1), (value, amount+2), ...

        # First, try increasing value (amount must be >= current amount)
        for value in range(last_bid.value + 1, 7):
            c2 = count_dice(hand2, value)
            # Try amounts from current amount up to what we can truthfully bid
            max_truthful = c2 + self.num_dice_per_player
            for amount in range(last_bid.amount, max_truthful + 1):
                bid = Bid(value=value, amount=amount)
                if is_legal_bid(bid, last_bid):
                    return bid

        # Then, try same value with increased amount
        c2 = count_dice(hand2, last_bid.value)
        max_truthful = c2 + self.num_dice_per_player
        for amount in range(last_bid.amount + 1, max_truthful + 1):
            bid = Bid(value=last_bid.value, amount=amount)
            if is_legal_bid(bid, last_bid):
                return bid

        return None  # No truthful bid found

    def _find_lowest_legal_bid(self, last_bid: Bid) -> Bid:
        """Find lowest legal bid (even if lying).

        Priority order: increase value first (with same or higher amount), then increase amount.
        When increasing value, amount must be >= current amount (per MATH.md).
        """
        # Try increasing value with minimum legal amount (same as current)
        for value in range(last_bid.value + 1, 7):
            bid = Bid(value=value, amount=last_bid.amount)
            if is_legal_bid(bid, last_bid):
                return bid

        # Then try same value with increased amount
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)
