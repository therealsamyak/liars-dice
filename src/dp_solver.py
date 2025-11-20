"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE
from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None,
    max_value: int = MAX_DICE_VALUE,
    max_amount: int = MAX_BID_AMOUNT,
) -> list[Bid]:
    """Generate all legal bids given the last bid.

    Bids are generated in priority order:
    1. First: increase face value (any quantity allowed, ordered by value then amount)
    2. Then: same face value, increase amount

    This ensures fixed bid order with priority: value first, then amount.
    Matches MATH.md: when increasing value, any amount is allowed.
    """
    if last_bid is None:
        # First bid: all possible bids, ordered by value then amount
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Priority 1: Increase value (any amount allowed when increasing value)
    # Ordered by value first, then amount (lowest first)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(1, max_amount + 1):  # Any amount allowed
            bids.append(Bid(value=value, amount=amount))

    # Priority 2: Same value, increase amount
    for amount in range(last_bid.amount + 1, max_amount + 1):
        bids.append(Bid(value=last_bid.value, amount=amount))

    return bids


def get_bid_position(
    bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
) -> int:
    """Get position of bid in total ordering of all possible bids.

    Bids are ordered by: value first, then amount.
    Position 0 = first possible bid (1,1)
    Position N-1 = last possible bid (6, max_amount)

    Returns:
        Position/index of bid in total ordering, or 0 if bid is None
    """
    if bid is None:
        return 0

    # Total ordering: (1,1), (1,2), ..., (1,max_amount), (2,1), (2,2), ..., (6,max_amount)
    # Position = (value-1) * max_amount + (amount-1)
    position = (bid.value - 1) * max_amount + (bid.amount - 1)
    return position


def get_max_bid_position(
    max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
) -> int:
    """Get total number of possible bids (max position + 1)."""
    return max_value * max_amount


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

    def __init__(
        self,
        num_dice_per_player: int,
        opponent_policy: TruthTellingPolicy,
        max_value: int = MAX_DICE_VALUE,
        max_amount: int = MAX_BID_AMOUNT,
    ):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(
            max_value, max_amount
        )  # Total number of possible bids

    def solve(self):
        """Solve using backward induction over all states."""
        print(f"Generating all possible hands ({self.num_hands} hands)...")

        # Group states by bid history depth for backward induction
        states_by_depth: dict[int, list[ObservableState]] = defaultdict(list)

        print("Enumerating all observable states...")
        all_visited = set()  # Track all states across all hands

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(
                    f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far..."
                )

            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            key = state_key(obs)
            if key not in all_visited:
                states_by_depth[0].append(obs)
                all_visited.add(key)
                # Recursively enumerate all reachable states from this hand
                enumeration_counter = {"calls": 0, "last_print": 0}
                self._enumerate_states(
                    obs,
                    states_by_depth,
                    max_depth=self.max_depth,
                    visited=all_visited,
                    enumeration_counter=enumeration_counter,
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

            for state_idx, obs in enumerate(states):
                if state_idx % 1000 == 0 and state_idx > 0:
                    print(
                        f"    Depth {depth}: {state_idx}/{len(states)} states computed..."
                    )
                self._compute_optimal_action(obs)

        print(f"DP solving complete! Total states: {len(self.policy)}")

    def _enumerate_states(
        self,
        obs: ObservableState,
        states_by_depth: dict[int, list[ObservableState]],
        max_depth: int,
        visited: set[str] | None = None,
        enumeration_counter: dict[str, int] | None = None,
    ):
        """Recursively enumerate all reachable states from a given state.

        Depth is based on position of last bid in total bid ordering,
        not just the number of bids in history.
        """
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

        # Depth = position of last bid in total ordering (or 0 if no bids)
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        depth = get_bid_position(last_bid, self.max_value, self.max_amount)

        if depth >= max_depth:
            return

        key = state_key(obs)
        # Avoid duplicates
        if key in visited:
            return
        visited.add(key)

        states_by_depth[depth].append(obs)

        # Enumerate states reachable by making a bid
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        legal_bids = generate_all_legal_bids(last_bid, self.max_value, self.max_amount)

        # For each legal bid, consider opponent's possible responses
        # Check ALL opponent hands to find all unique reachable states
        seen_next_states = set()

        for bid_idx, bid in enumerate(legal_bids):
            new_bid_history = obs.bid_history + [bid]

            # Check all opponent hands to find unique responses
            unique_opponent_bids = set()
            for hand_idx, hand2_tuple in enumerate(self.all_hands):
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal outcome, not a new state
                    continue
                else:
                    # Track unique opponent bid (value, amount) pairs
                    unique_opponent_bids.add(
                        (opponent_action.value, opponent_action.amount)
                    )

                # Progress tracking: print every 10000 opponent hand checks
                enumeration_counter["calls"] += 1
                if (
                    enumeration_counter["calls"] - enumeration_counter["last_print"]
                    >= 10000
                ):
                    total_states = sum(
                        len(states) for states in states_by_depth.values()
                    )
                    print(
                        f"    Enumerating: {enumeration_counter['calls']} opponent checks, {total_states} states found..."
                    )
                    enumeration_counter["last_print"] = enumeration_counter["calls"]

            # Create states for each unique opponent bid
            for opp_value, opp_amount in unique_opponent_bids:
                opponent_bid_history = new_bid_history + [Bid(opp_value, opp_amount)]
                next_key = state_key(
                    ObservableState(
                        hand1=obs.hand1.copy(),
                        bid_history=opponent_bid_history,
                        game_status=GameStatus.ACTIVE,
                    )
                )
                if next_key not in seen_next_states and next_key not in visited:
                    seen_next_states.add(next_key)
                    new_obs = ObservableState(
                        hand1=obs.hand1.copy(),
                        bid_history=opponent_bid_history,
                        game_status=GameStatus.ACTIVE,
                    )
                    # Recursively enumerate from this state
                    self._enumerate_states(
                        new_obs,
                        states_by_depth,
                        max_depth,
                        visited,
                        enumeration_counter,
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
        legal_bids = generate_all_legal_bids(last_bid, self.max_value, self.max_amount)

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
