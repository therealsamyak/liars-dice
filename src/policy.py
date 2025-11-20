"""Optimal policy implementation for Player 1."""

import json
import os
from collections import Counter

from parameters import LIAR_THRESHOLD
from src.game_state import Bid, ObservableState
from src.probability import p_valid


def state_key(obs: ObservableState) -> str:
    """Create a unique key for an observable state.

    Note: Hand is sorted to ensure same dice counts map to same key.
    Must match state_key in dp_solver.py.
    """
    # Sort hand to normalize (order doesn't matter, only counts)
    hand_str = ",".join(map(str, sorted(obs.hand1)))
    bid_str = "|".join(f"{b.value},{b.amount}" for b in obs.bid_history)
    status_str = obs.game_status.value
    return f"{hand_str};{bid_str};{status_str}"


class OptimalPolicy:
    """Optimal policy for Player 1 using precomputed DP policy."""

    def __init__(self, num_dice_per_player: int, policy_file: str | None = None):
        self.num_dice_per_player = num_dice_per_player
        self.policy: dict[str, dict | None] = {}

        # Default policy file path
        if policy_file is None:
            policy_file = os.path.join(
                "policy", f"policy_{num_dice_per_player}dice.json"
            )

        self.policy_file = policy_file

        # Load policy if file exists
        if os.path.exists(policy_file):
            self.load_policy(policy_file)
        else:
            print(
                f"Warning: Policy file {policy_file} not found. Using heuristic fallback."
            )
            self.policy = {}

    def load_policy(self, filepath: str):
        """Load policy from JSON file."""
        with open(filepath, "r") as f:
            policy_dict = json.load(f)

        self.policy = {}
        for key, action in policy_dict.items():
            if action is None:
                self.policy[key] = None
            else:
                self.policy[key] = action

        # print(f"Loaded policy from {filepath} ({len(self.policy)} states)")

    def get_action(self, obs: ObservableState) -> Bid | None:
        """Returns Bid to make or None to call LIAR."""
        if obs.game_status.value != "Active":
            return None

        # Try to look up action in precomputed policy
        key = state_key(obs)
        if key in self.policy:
            action = self.policy[key]
            if action is None:
                return None  # Call LIAR
            else:
                return Bid(value=action["value"], amount=action["amount"])

        # Fallback to heuristic if policy not found
        return self._heuristic_action(obs)

    def _heuristic_action(self, obs: ObservableState) -> Bid | None:
        """Heuristic fallback when policy not found."""
        last_bid = obs.bid_history[-1] if obs.bid_history else None

        # If no previous bid, make a conservative first bid
        if last_bid is None:
            counts = Counter(obs.hand1)
            if counts:
                most_common_value, most_common_count = counts.most_common(1)[0]
                expected_opponent = self.num_dice_per_player // 6
                amount = most_common_count + max(1, expected_opponent)
                return Bid(value=most_common_value, amount=amount)
            else:
                return Bid(value=1, amount=1)

        # Calculate probability that the last bid is valid
        prob_valid = p_valid(last_bid, obs.hand1, self.num_dice_per_player)

        # If probability is very low, call LIAR
        if prob_valid < LIAR_THRESHOLD:
            return None

        # Otherwise, make a higher bid
        if last_bid.value < 6:
            new_value = last_bid.value + 1
            c1 = sum(1 for die in obs.hand1 if die == new_value)
            expected_opponent = self.num_dice_per_player // 6
            new_amount = c1 + max(1, expected_opponent)
            if new_value > last_bid.value or (
                new_value == last_bid.value and new_amount > last_bid.amount
            ):
                return Bid(value=new_value, amount=new_amount)

        # Increase amount on same value
        new_amount = last_bid.amount + 1
        return Bid(value=last_bid.value, amount=new_amount)
