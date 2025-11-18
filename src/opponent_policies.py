"""Opponent policy implementations."""

from collections import Counter

from src.game_state import Bid
from src.rules import count_dice, is_legal_bid


class TruthTellingPolicy:
    """Opponent policy that bids truthfully when possible."""

    def __init__(self, num_dice_per_player: int):
        self.num_dice_per_player = num_dice_per_player

    def get_action(self, hand2: list[int], bid_history: list[Bid]) -> Bid | None:
        """Returns Bid to make or None to call LIAR."""
        last_bid = bid_history[-1] if bid_history else None

        # If no previous bid, make a truthful first bid
        if last_bid is None:
            # Count opponent's dice and make a conservative bid
            counts = Counter(hand2)
            if counts:
                most_common_value, most_common_count = counts.most_common(1)[0]
                # Bid what we have plus expected from opponent
                expected_opponent = self.num_dice_per_player // 6
                amount = most_common_count + max(1, expected_opponent)
                return Bid(value=most_common_value, amount=amount)
            else:
                return Bid(value=1, amount=1)

        # First, check if the last bid is actually impossible (definitely invalid)
        # A truth-telling opponent only calls LIAR if the bid is impossible
        # The bid is impossible if: our dice + maximum possible from opponent < bid amount
        c2_last = count_dice(hand2, last_bid.value)
        max_possible_from_opponent = (
            self.num_dice_per_player
        )  # Opponent could have all dice of this value

        if c2_last + max_possible_from_opponent < last_bid.amount:
            # Last bid is impossible - call LIAR
            return None

        # Last bid is possible, try to find a truthful higher bid
        # Strategy: Try all possible legal bids, prioritizing truthful ones
        expected_opponent = self.num_dice_per_player // 6

        # First, try all truthful bids (bids we can verify are true)
        # Try increasing value first (preferred strategy)
        for value in range(last_bid.value + 1, 7):
            c2 = count_dice(hand2, value)
            # When increasing value, minimum legal amount is 1
            # Try amounts from 1 up to what we can truthfully bid
            max_truthful_amount = c2 + expected_opponent

            for amount in range(1, max_truthful_amount + 1):
                new_bid = Bid(value=value, amount=amount)
                if is_legal_bid(new_bid, last_bid):
                    # This is a truthful bid (we expect enough dice)
                    total_expected = c2 + expected_opponent
                    if total_expected >= amount:
                        return new_bid

        # Try increasing amount on same value (if we can do it truthfully)
        if last_bid.value <= 6:
            c2 = count_dice(hand2, last_bid.value)
            new_amount = last_bid.amount + 1
            total_expected = c2 + expected_opponent
            if total_expected >= new_amount:
                new_bid = Bid(value=last_bid.value, amount=new_amount)
                if is_legal_bid(new_bid, last_bid):
                    return new_bid

        # No truthful legal bid found - try to find ANY legal bid
        # This is a last resort - we'll make a bid that might be untruthful
        # but is still legal according to game rules

        # Try increasing value with minimal legal amount
        for value in range(last_bid.value + 1, 7):
            min_amount = 1 if value > last_bid.value else last_bid.amount + 1
            new_bid = Bid(value=value, amount=min_amount)
            if is_legal_bid(new_bid, last_bid):
                return new_bid

        # Last resort: increase amount on same value
        # This is the minimal legal bid, even if untruthful
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)
