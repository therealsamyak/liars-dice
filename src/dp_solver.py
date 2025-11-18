"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
) -> list[Bid]:
    """Generate all legal bids given the last bid.

    Bids are generated in priority order:
    1. First: increase face value (amount >= current amount, ordered by value then amount)
    2. Then: same face value, increase amount

    This ensures fixed bid order with priority: value first, then amount.
    Matches MATH.md: when increasing value, amount must be >= current amount.
    """
    if last_bid is None:
        # First bid: all possible bids, ordered by value then amount
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Priority 1: Increase value (amount must be >= current amount)
    # Ordered by value first, then amount (lowest first)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(
            last_bid.amount, max_amount + 1
        ):  # Start from current amount
            bids.append(Bid(value=value, amount=amount))

    # Priority 2: Same value, increase amount
    for amount in range(last_bid.amount + 1, max_amount + 1):
        bids.append(Bid(value=last_bid.value, amount=amount))

    return bids


def state_key(obs: ObservableState) -> str:
    """Create a unique key for an observable state.

    Note: Hand is sorted to ensure same dice counts map to same key.
    Must match state_key in policy.py.
    """
    # Sort hand to normalize (order doesn't matter, only counts)
    hand_str = ",".join(map(str, sorted(obs.hand1)))
    bid_str = "|".join(f"{b.value},{b.amount}" for b in obs.bid_history)
    status_str = obs.game_status.value
    return f"{hand_str};{bid_str};{status_str}"


def compute_reward(state: State, caller: int, last_bid: Bid) -> float:
    """Compute reward for terminal state.

    Returns:
        +1.0 if Player 1 wins
        -1.0 if Player 2 wins
        0.0 otherwise
    """
    winner = check_win_condition(state, caller, last_bid)
    if winner == 1:
        return 1.0
    elif winner == 2:
        return -1.0
    return 0.0


class DPSolver:
    """DP solver that enumerates all states and computes optimal policy.

    Uses backward induction: processes states by bid history depth,
    computing optimal actions from terminal states backwards.
    """

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior

    def solve(self):
        """Solve using backward induction over all states."""
        print(f"Generating all possible hands ({self.num_hands} hands)...")

        # Group states by bid history depth for backward induction
        states_by_depth: dict[int, list[ObservableState]] = defaultdict(list)

        print("Enumerating all observable states...")
        all_visited = set()  # Track all states across all hands
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            key = state_key(obs)
            if key not in all_visited:
                states_by_depth[0].append(obs)
                all_visited.add(key)
                # Recursively enumerate all reachable states from this hand
                self._enumerate_states(
                    obs, states_by_depth, max_depth=30, visited=all_visited
                )

        print(
            f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}"
        )

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(
            f"Computing optimal policy using backward induction (max depth: {max_depth})..."
        )

        for depth in range(max_depth, -1, -1):
            if depth not in states_by_depth:
                continue

            states = states_by_depth[depth]
            print(f"  Processing depth {depth}: {len(states)} states")

            for obs in states:
                self._compute_optimal_action(obs)

        print(f"DP solving complete! Total states: {len(self.policy)}")

    def _enumerate_states(
        self,
        obs: ObservableState,
        states_by_depth: dict[int, list[ObservableState]],
        max_depth: int,
        visited: set[str] | None = None,
    ):
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()

        depth = len(obs.bid_history)
        if depth > max_depth:
            return

        key = state_key(obs)
        # Avoid duplicates
        if key in visited:
            return
        visited.add(key)

        states_by_depth[depth].append(obs)

        # Enumerate states reachable by making a bid
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        legal_bids = generate_all_legal_bids(last_bid)

        # For each legal bid, consider opponent's possible responses
        # We need to consider all possible opponent responses (they depend on opponent's hand)
        # So we'll enumerate states that could be reached
        seen_next_states = set()
        for bid in legal_bids:
            new_bid_history = obs.bid_history + [bid]

            # Consider all possible opponent hands to determine their response
            for hand2_tuple in self.all_hands:
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - this is a terminal outcome, not a new state
                    # The value will be computed in _compute_bid_value
                    continue
                else:
                    # Opponent makes a bid - create new observable state
                    opponent_bid_history = new_bid_history + [opponent_action]
                    next_key = state_key(
                        ObservableState(
                            hand1=obs.hand1.copy(),
                            bid_history=opponent_bid_history,
                            game_status=GameStatus.ACTIVE,
                        )
                    )
                    if next_key not in seen_next_states:
                        seen_next_states.add(next_key)
                        new_obs = ObservableState(
                            hand1=obs.hand1.copy(),
                            bid_history=opponent_bid_history,
                            game_status=GameStatus.ACTIVE,
                        )
                        # Recursively enumerate from this state
                        self._enumerate_states(
                            new_obs, states_by_depth, max_depth, visited
                        )

    def _compute_optimal_action(self, obs: ObservableState) -> float:
        """Compute optimal action and value for a state using backward induction."""
        key = state_key(obs)

        # If already computed, return cached value
        if key in self.value_function:
            return self.value_function[key]

        # If game is over, value is 0 (shouldn't happen in our enumeration)
        if obs.game_status == GameStatus.GAME_OVER:
            self.value_function[key] = 0.0
            self.policy[key] = None
            return 0.0

        best_value = float("-inf")
        best_action = None

        # Action 1: Call LIAR (if there's a bid to challenge)
        if obs.bid_history:
            liar_value = self._compute_liar_value(obs)
            if liar_value > best_value:
                best_value = liar_value
                best_action = None  # None means call LIAR

        # Action 2: Make a bid
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        legal_bids = generate_all_legal_bids(last_bid)

        for bid in legal_bids:
            bid_value = self._compute_bid_value(obs, bid)
            if bid_value > best_value:
                best_value = bid_value
                best_action = {"value": bid.value, "amount": bid.amount}

        # Store computed value and policy
        self.value_function[key] = best_value
        self.policy[key] = best_action

        return best_value

    def _compute_liar_value(self, obs: ObservableState) -> float:
        """Compute expected value of calling LIAR, marginalizing over opponent's hand."""
        if not obs.bid_history:
            return float("-inf")

        last_bid = obs.bid_history[-1]
        total_value = 0.0

        # Marginalize over all possible opponent hands (uniform prior)
        for hand2_tuple in self.all_hands:
            hand2 = list(hand2_tuple)

            # Create full state
            state = State(
                hand1=obs.hand1.copy(),
                hand2=hand2,
                bid_history=obs.bid_history.copy(),
                game_status=GameStatus.GAME_OVER,
            )

            # Compute reward
            reward = compute_reward(state, caller=1, last_bid=last_bid)
            total_value += reward * self.hand_prob

        return total_value

    def _compute_bid_value(self, obs: ObservableState, bid: Bid) -> float:
        """Compute expected value of making a bid, marginalizing over opponent's hand."""
        new_bid_history = obs.bid_history + [bid]
        total_value = 0.0

        # Marginalize over all possible opponent hands
        for hand2_tuple in self.all_hands:
            hand2 = list(hand2_tuple)

            # Get opponent's action given their hand and new bid history
            opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

            if opponent_action is None:
                # Opponent calls LIAR - terminal state
                state = State(
                    hand1=obs.hand1.copy(),
                    hand2=hand2,
                    bid_history=new_bid_history.copy(),
                    game_status=GameStatus.GAME_OVER,
                )
                reward = compute_reward(state, caller=2, last_bid=bid)
                total_value += reward * self.hand_prob
            else:
                # Opponent makes a bid - create new observable state
                opponent_bid_history = new_bid_history + [opponent_action]
                new_obs = ObservableState(
                    hand1=obs.hand1.copy(),
                    bid_history=opponent_bid_history,
                    game_status=GameStatus.ACTIVE,
                )

                # Recursively compute value of this new state
                # (should already be computed due to backward induction)
                value = self._compute_optimal_action(new_obs)
                total_value += value * self.hand_prob

        return total_value

    def save_policy(self, filepath: str):
        """Save computed policy to JSON file."""
        policy_dict = {}
        for key, action in self.policy.items():
            policy_dict[key] = action

        with open(filepath, "w") as f:
            json.dump(policy_dict, f, indent=2)

        print(f"Policy saved to {filepath} ({len(policy_dict)} states)")

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

        print(f"Policy loaded from {filepath} ({len(self.policy)} states)")
