# Implementing dp type algorithm for bot actions

_Exported on 11/20/2025 at 11:15:31 PST from Cursor (2.0.77)_

---

**User**

we did this wrong @MATH.md gives us system dynamics, we need to run a 'dp type' algorithm that goes over every single state and gives our 'bot' an optimal action to take, then the simulation takes that 'dp matrix' (or .json file) to determine what action to tkae

---

**Cursor**

Reviewing the codebase to understand the current structure before implementing a DP algorithm that precomputes optimal actions for all states.

Creating a DP solver that precomputes optimal actions for all observable states. Implementing the solver:

Reading `dp_solver.py`:

Rewriting the DP solver to systematically enumerate all states and compute optimal actions. Checking how it's currently used:

Rewriting the DP solver to systematically enumerate all states and compute optimal actions via backward induction:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition, count_total_dice, is_legal_bid


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
) -> list[Bid]:
    """Generate all legal bids given the last bid."""
    if last_bid is None:
        # First bid: all possible bids
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Increase value (any amount >= 1)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(1, max_amount + 1):
            bids.append(Bid(value=value, amount=amount))

    # Same value, increase amount
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
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            states_by_depth[0].append(obs)

            # Recursively enumerate all reachable states from this hand
            self._enumerate_states(obs, states_by_depth, max_depth=30)

        print(f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}")

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(f"Computing optimal policy using backward induction (max depth: {max_depth})...")

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
    ):
        """Recursively enumerate all reachable states from a given state."""
        depth = len(obs.bid_history)
        if depth > max_depth:
            return

        key = state_key(obs)
        # Avoid duplicates
        existing_states = {state_key(s) for s in states_by_depth[depth]}
        if key in existing_states:
            return

        states_by_depth[depth].append(obs)

        # Enumerate states reachable by making a bid
        last_bid = obs.bid_history[-1] if obs.bid_history else None
        legal_bids = generate_all_legal_bids(last_bid)

        # For each legal bid, consider opponent's possible responses
        for bid in legal_bids:
            new_bid_history = obs.bid_history + [bid]

            # Consider all possible opponent hands to determine their response
            opponent_responses = set()
            for hand2_tuple in self.all_hands:
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

                if opponent_action is None:
                    # Opponent calls LIAR - terminal state
                    continue
                else:
                    # Opponent makes a bid - create new observable state
                    opponent_bid_history = new_bid_history + [opponent_action]
                    new_obs = ObservableState(
                        hand1=obs.hand1.copy(),
                        bid_history=opponent_bid_history,
                        game_status=GameStatus.ACTIVE,
                    )
                    # Recursively enumerate from this state
                    self._enumerate_states(new_obs, states_by_depth, max_depth)

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

```

Updating `main.py` to generate the policy if missing, and ensuring the simulation uses the precomputed policy:

Verifying `state_key` matches between `dp_solver.py` and `policy.py`:

Comparing the implementations to ensure they match:

Updating `main.py` to check for a policy file and provide guidance:

```python
"""Main entry point for Liar's Dice simulation."""

import os

from src.simulation import GameSimulation


def main():
    """Run the Liar's Dice simulation with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = 5
    policy_file = f"policy_{num_dice}dice.json"

    # Check if policy file exists
    if not os.path.exists(policy_file):
        print(
            f"Warning: Policy file '{policy_file}' not found.\n"
            f"Run 'python generate_policy.py' first to generate optimal policy.\n"
            f"Using heuristic fallback for now.\n"
        )

    # Create simulation with default 5 dice per player
    simulation = GameSimulation(num_dice_per_player=num_dice)

    # Run a single game
    winner, final_state = simulation.simulate_game(verbose=True)

    print("\n=== Game Over ===")
    print(f"Final winner: Player {winner}")
    print(f"Total bids made: {len(final_state.bid_history)}")


if __name__ == "__main__":
    main()

```

Fixing the DP solver: the enumeration approach may miss states. Switching to a more systematic method:

Fixing state enumeration: terminal states (when opponent calls LIAR) aren't being created. Updating the solver:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition, count_total_dice, is_legal_bid


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
) -> list[Bid]:
    """Generate all legal bids given the last bid."""
    if last_bid is None:
        # First bid: all possible bids
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Increase value (any amount >= 1)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(1, max_amount + 1):
            bids.append(Bid(value=value, amount=amount))

    # Same value, increase amount
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
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            states_by_depth[0].append(obs)

            # Recursively enumerate all reachable states from this hand
            self._enumerate_states(obs, states_by_depth, max_depth=30)

        print(f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}")

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(f"Computing optimal policy using backward induction (max depth: {max_depth})...")

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
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

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
                        self._enumerate_states(new_obs, states_by_depth, max_depth, visited)

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

```

Checking for lint errors and verifying the implementation:

Fixing a bug in state enumeration: the visited check isn't working correctly. Updating it:

Simplifying and fixing the enumeration to share the visited set correctly:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition, count_total_dice, is_legal_bid


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
) -> list[Bid]:
    """Generate all legal bids given the last bid."""
    if last_bid is None:
        # First bid: all possible bids
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Increase value (any amount >= 1)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(1, max_amount + 1):
            bids.append(Bid(value=value, amount=amount))

    # Same value, increase amount
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
                self._enumerate_states(obs, states_by_depth, max_depth=30, visited=all_visited)

        print(f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}")

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(f"Computing optimal policy using backward induction (max depth: {max_depth})...")

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
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

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
                        self._enumerate_states(new_obs, states_by_depth, max_depth, visited)

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

```

## Summary

Refactored the codebase to use a DP approach that precomputes optimal actions for all observable states.

### Changes Made:

1. **Rewrote `src/dp_solver.py`**:
   - Enumerates all observable states (H1, D, Go) systematically
   - Uses backward induction: processes states by bid history depth, computing optimal actions from deeper states backwards
   - Marginalizes over opponent's unknown hand H2 using uniform prior
   - Computes expected values for all actions (bids and calling LIAR)
   - Saves the computed policy to a JSON file

2. **Updated `main.py`**:
   - Added a check for the policy file
   - Provides guidance to run `generate_policy.py` first if the policy is missing

### How It Works:

1. **Generate Policy**: Run `python generate_policy.py` to:
   - Enumerate all observable states reachable from any initial hand
   - Compute optimal actions using backward induction
   - Save the policy to `policy_{N}dice.json`

2. **Run Simulation**: Run `python main.py` to:
   - Load the precomputed policy from JSON
   - Use it to determine optimal actions during simulation

The DP solver:

- Enumerates states by simulating all possible opponent responses (considering all possible opponent hands)
- Processes states in reverse depth order (backward induction)
- For each state, computes the expected value of calling LIAR vs. making each legal bid
- Stores the optimal action (bid or call LIAR) for each state

The simulation now uses the precomputed DP matrix (JSON file) instead of computing actions on the fly.

---

**User**

also liar's dice has a fixed bid order with priority (can't go lower than current bid), so incorperate that

for the opponent policy, its legal bid should be the lowest bid it can do that is truthful, otherwise lie with the lowest bid possible

---

**Cursor**

Reviewing the current bid ordering and opponent policy to incorporate fixed bid priority and update the opponent to prefer the lowest truthful legal bid, otherwise the lowest legal bid.

Updating the opponent policy to prefer the lowest truthful legal bid, otherwise the lowest legal bid. Verifying bid ordering is correct:

Updating the opponent policy and verifying bid ordering:

```python
"""Opponent policy implementations."""

from src.game_state import Bid
from src.rules import count_dice, count_total_dice, is_legal_bid


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
        """
        # Try bids in priority order: increase value first, then increase amount
        # Priority: (value+1, 1), (value+1, 2), ..., (6, 1), (6, 2), ...
        #          then (value, amount+1), (value, amount+2), ...

        # First, try increasing value (any amount >= 1)
        for value in range(last_bid.value + 1, 7):
            c2 = count_dice(hand2, value)
            # Try amounts from 1 up to what we can truthfully bid
            max_truthful = c2 + self.num_dice_per_player
            for amount in range(1, max_truthful + 1):
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

        Priority order: increase value first, then increase amount.
        """
        # Try increasing value with minimum amount (1)
        for value in range(last_bid.value + 1, 7):
            bid = Bid(value=value, amount=1)
            if is_legal_bid(bid, last_bid):
                return bid

        # Then try same value with increased amount
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)

```

Removing the unused import and verifying bid ordering in the DP solver:

```python
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
        """
        # Try bids in priority order: increase value first, then increase amount
        # Priority: (value+1, 1), (value+1, 2), ..., (6, 1), (6, 2), ...
        #          then (value, amount+1), (value, amount+2), ...

        # First, try increasing value (any amount >= 1)
        for value in range(last_bid.value + 1, 7):
            c2 = count_dice(hand2, value)
            # Try amounts from 1 up to what we can truthfully bid
            max_truthful = c2 + self.num_dice_per_player
            for amount in range(1, max_truthful + 1):
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

        Priority order: increase value first, then increase amount.
        """
        # Try increasing value with minimum amount (1)
        for value in range(last_bid.value + 1, 7):
            bid = Bid(value=value, amount=1)
            if is_legal_bid(bid, last_bid):
                return bid

        # Then try same value with increased amount
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)

```

Verifying the DP solver uses bid ordering correctly and updating it to respect priority:

---

**User**

also in terms of the math, we could be wrong about the rules of liar's dice bidding so please correct that

---

**Cursor**

Reviewing the bidding rules and comparing them with standard Liar's Dice rules:

Updating the bidding rules to match standard Liar's Dice: a bid must increase either the face value or the quantity (or both). Verifying and fixing the implementation:

Verifying the bidding rules match the MATH.md examples. The current implementation allows increasing face value (with any amount) or increasing amount (same face), which matches the examples. Ensuring the priority ordering is correct and updating documentation:

```python
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

    Bidding rules with fixed priority order:
    - Face value has priority over amount
    - You can increase face value (amount can be any value >= 1)
    - OR you can keep same face value and increase amount
    - You cannot decrease face value
    - You cannot decrease amount when keeping same face value

    Examples:
    - "three 3s" → "two 4s" valid (increase value, decrease amount OK)
    - "four 2s" → "two 3s" valid (increase value, decrease amount OK)
    - "four 4s" → "three 3s" invalid (decrease both)
    - "three 3s" → "four 3s" valid (same value, increase amount)
    - "three 3s" → "two 3s" invalid (same value, decrease amount)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current (amount can be any >= 1)
        True  if DiceValue_new = DiceValue_current and Amount_new > Amount_current
        False otherwise
    }
    """
    if D_last is None:
        # D = ∅: first bid - any valid bid is allowed
        return True

    # DiceValue_new > DiceValue_current
    # When increasing value, amount can be any value >= 1 (can decrease amount)
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
    WIN(s): Determine the winner after a LIAR call.

    WIN(s) = {
        P1  if a = C, G_o' = GameOver, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        P2  if a = C, G_o' = GameOver, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P1  if opponent calls liar, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P2  if opponent calls liar, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        None if G_o' = Active
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

```

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition, count_total_dice, is_legal_bid


def generate_all_hands(num_dice: int) -> list[tuple[int, ...]]:
    """Generate all possible dice hands of length num_dice."""
    return list(product(range(1, 7), repeat=num_dice))


def generate_all_legal_bids(
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
) -> list[Bid]:
    """Generate all legal bids given the last bid.

    Bids are generated in priority order:
    1. First: increase face value (any amount >= 1, ordered by value then amount)
    2. Then: same face value, increase amount

    This ensures fixed bid order with priority: value first, then amount.
    """
    if last_bid is None:
        # First bid: all possible bids, ordered by value then amount
        bids = []
        for value in range(1, max_value + 1):
            for amount in range(1, max_amount + 1):
                bids.append(Bid(value=value, amount=amount))
        return bids

    bids = []
    # Priority 1: Increase value (any amount >= 1)
    # Ordered by value first, then amount (lowest first)
    for value in range(last_bid.value + 1, max_value + 1):
        for amount in range(1, max_amount + 1):
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
                self._enumerate_states(obs, states_by_depth, max_depth=30, visited=all_visited)

        print(f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}")

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(f"Computing optimal policy using backward induction (max depth: {max_depth})...")

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
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

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
                        self._enumerate_states(new_obs, states_by_depth, max_depth, visited)

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

```

Updating MATH.md to clarify the bidding rules:

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules with fixed priority order:**
  - Face value has priority over amount
  - You can increase face value (amount can be any value >= 1, even lower than current)
  - OR you can keep same face value and increase amount
  - You cannot decrease face value
  - You cannot decrease amount when keeping same face value
  - Examples: three 3s $\rightarrow$ two 4s valid (increase value, decrease amount OK), four 2s $\rightarrow$ two 3s valid (increase value, decrease amount OK), four 4s $\rightarrow$ three 3s invalid (decrease both), three 3s $\rightarrow$ four 3s valid (same value, increase amount)
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

### Math Formulations

$S = (H_1, H_2, D, G_o)$ - state space

- $H_1, H_2 \in \{1,2,3,4,5,6\}^N$ - dice hands for player 1 and player 2
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history $D_{last}$ - latest bid
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$O = (H_1, D, G_o)$ - observable space

- $H_1 \in \{1, 2, 3, 4, 5, 6\}^N$ - our player's dice hand
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$A \in \{B, C\}$ - Action set

- $B = (\text{DiceValue},\text{Amount})$ - new bid
- $C$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($G_o = \text{GameOver}$)

$T(s' \mid s, A)$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}$

Where:
$\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

- Hands $H_1, H_2$ remain constant throughout a single game. Only $D$ and $G_o$ change.

New observation: $\Omega(o \mid s', a) = (H_1', D', G_o')$
<br>

Reward: $R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$

Win Conditions: $WIN(s) = \begin{cases}
\text{P1} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o' = \text{Active}
\end{cases}$

Where $\text{COUNT}(H, v)$ returns the number of dice in hand $H$ showing value $v$, and $D_{\text{last}}$ is the most recent bid.

$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$ - baseline

- Any bid given from opponent is not valid proof of the hand state.

HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $b(H_2 \mid H_1, D_last)$

$D_last = (v*, a*)$ - value, amount

$P_{valid}(D_{last} \mid H_1) = P(COUNT(H_1 \cup H_2, v*) \ge a* | H_1)$

$c_1 = COUNT(H_1, v*)$, $c_2 = COUNT(H_2, v*)$ - number of dice in each hand with value $v*$
$k = a* - c_1$ - additional dice needed from opponent for bid to be valid

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k)$

Because we don't know opponent's hand / policy, $c_2 ~ Binomial(N, p = \frac{1}{6}$

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$

- If $k > N$, then $P_{valid} = 0$ - impossible bid
- If $k \le 0$, then $P_{valid} = 1$ - our hand satisfies bid

### Notes:

- Modeled from Player 1 perspective. Opponent actions handled by system internally.
```

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules with fixed priority order:**
  - Face value has priority over amount
  - You can increase face value (amount can be any value >= 1, even lower than current)
  - OR you can keep same face value and increase amount
  - You cannot decrease face value
  - You cannot decrease amount when keeping same face value
  - Examples: three 3s $\rightarrow$ two 4s valid (increase value, decrease amount OK), four 2s $\rightarrow$ two 3s valid (increase value, decrease amount OK), four 4s $\rightarrow$ three 3s invalid (decrease both), three 3s $\rightarrow$ four 3s valid (same value, increase amount)
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

### Math Formulations

$S = (H_1, H_2, D, G_o)$ - state space

- $H_1, H_2 \in \{1,2,3,4,5,6\}^N$ - dice hands for player 1 and player 2
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history $D_{last}$ - latest bid
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$O = (H_1, D, G_o)$ - observable space

- $H_1 \in \{1, 2, 3, 4, 5, 6\}^N$ - our player's dice hand
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$A \in \{B, C\}$ - Action set

- $B = (\text{DiceValue},\text{Amount})$ - new bid
- $C$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($G_o = \text{GameOver}$)

$T(s' \mid s, A)$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}$

Where:
$\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ (amount can be any } \ge 1\text{)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

**Bid Priority Order:** When generating legal bids, priority is:

1. First: increase face value (ordered by value then amount, lowest first)
2. Then: same face value, increase amount

- Hands $H_1, H_2$ remain constant throughout a single game. Only $D$ and $G_o$ change.

New observation: $\Omega(o \mid s', a) = (H_1', D', G_o')$
<br>

Reward: $R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$

Win Conditions: $WIN(s) = \begin{cases}
\text{P1} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o' = \text{Active}
\end{cases}$

Where $\text{COUNT}(H, v)$ returns the number of dice in hand $H$ showing value $v$, and $D_{\text{last}}$ is the most recent bid.

$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$ - baseline

- Any bid given from opponent is not valid proof of the hand state.

HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $b(H_2 \mid H_1, D_last)$

$D_last = (v*, a*)$ - value, amount

$P_{valid}(D_{last} \mid H_1) = P(COUNT(H_1 \cup H_2, v*) \ge a* | H_1)$

$c_1 = COUNT(H_1, v*)$, $c_2 = COUNT(H_2, v*)$ - number of dice in each hand with value $v*$
$k = a* - c_1$ - additional dice needed from opponent for bid to be valid

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k)$

Because we don't know opponent's hand / policy, $c_2 ~ Binomial(N, p = \frac{1}{6}$

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$

- If $k > N$, then $P_{valid} = 0$ - impossible bid
- If $k \le 0$, then $P_{valid} = 1$ - our hand satisfies bid

### Notes:

- Modeled from Player 1 perspective. Opponent actions handled by system internally.
```

## Summary of Changes

1. **Updated Opponent Policy** (`src/opponent_policies.py`):
   - Finds the lowest truthful legal bid (can be verified as true given their hand)
   - If no truthful bid exists, finds the lowest legal bid (even if lying)
   - Respects bid priority: increase value first, then increase amount

2. **Verified and Documented Bid Ordering**:
   - Face value has priority over amount
   - You can increase face value (amount can be any >= 1, even lower than current)
   - OR you can keep same face value and increase amount
   - You cannot decrease face value
   - You cannot decrease amount when keeping same face value
   - Bids are generated in priority order: value first (lowest first), then amount

3. **Updated Documentation**:
   - Added examples in `src/rules.py` showing valid/invalid bids
   - Updated `MATH.md` to clarify the bidding rules
   - Added comments explaining the priority ordering

The implementation matches the MATH.md examples:

- "three 3s" → "two 4s" ✓ valid (increase value, decrease amount OK)
- "four 2s" → "two 3s" ✓ valid (increase value, decrease amount OK)
- "four 4s" → "three 3s" ✗ invalid (decrease both)
- "three 3s" → "four 3s" ✓ valid (same value, increase amount)

The opponent policy now finds the lowest legal bid that is truthful, otherwise the lowest legal bid, respecting the fixed priority order.

---

**User**

no the math.md itself could be completely wrong thats what i want you to fix

---

**Cursor**

Reviewing MATH.md against the implementation and standard Liar's Dice rules to identify and fix any errors.

Analyzing MATH.md for errors and inconsistencies:

---

**User**

math.md is the source of truth for everything. compare its rules to the liar's dice official rules, then fix math.md

then, fix the code to follow math.md

---

**Cursor**

Comparing MATH.md to official Liar's Dice rules, then updating MATH.md and aligning the code.

Updating MATH.md to match official Liar's Dice rules, then aligning the code.

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value (or both)
  - You can increase quantity while keeping same face value (e.g., "three 3s" → "four 3s")
  - You can increase face value while keeping same quantity (e.g., "three 3s" → "three 4s")
  - You can increase both quantity and face value (e.g., "three 3s" → "four 4s")
  - You CANNOT decrease quantity when increasing face value (e.g., "three 3s" → "two 4s" is INVALID)
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "four 4s" ✓ valid (increase both)
    - "three 3s" → "two 4s" ✗ invalid (increase value but decrease quantity)
    - "four 4s" → "three 3s" ✗ invalid (decrease both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
- At any point, instead of bidding, a player can call 'LIAR!', ending the game.
  - If the bid is invalid (COUNT < Amount), the caller wins
  - If the bid is valid (COUNT >= Amount), the bidder wins

### Math Formulations

$S = (H_1, H_2, D, G_o)$ - state space

- $H_1, H_2 \in \{1,2,3,4,5,6\}^N$ - dice hands for player 1 and player 2
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history $D_{last}$ - latest bid
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$O = (H_1, D, G_o)$ - observable space

- $H_1 \in \{1, 2, 3, 4, 5, 6\}^N$ - our player's dice hand
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$A \in \{B, C\}$ - Action set

- $B = (\text{DiceValue},\text{Amount})$ - new bid
- $C$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($G_o = \text{GameOver}$)

$T(s' \mid s, a)$ - transition probabilities

- If action is bid $B$: $H_1' = H_1, H_2' = H_2, D' = D \cup \{B\}, G_o' = \text{Active}$ (if $B \in \text{LEGAL}(D)$)
- If action is call LIAR $C$: $H_1' = H_1, H_2' = H_2, D' = D, G_o' = \text{GameOver}$

$T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, D' = D, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}$

Where:
$\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} \ge \text{Amount}_{\text{current}} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

**Bid Priority Order:** When generating legal bids, priority is:

1. First: increase face value (keeping same or higher quantity, ordered by value then amount, lowest first)
2. Then: same face value, increase amount

- Hands $H_1, H_2$ remain constant throughout a single game. Only $D$ and $G_o$ change.

New observation: $\Omega(o \mid s', a) = (H_1', D', G_o')$

Reward: $R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$

Win Conditions: $WIN(s, caller)$ where $caller \in \{1, 2\}$ is the player who called LIAR:

$WIN(s, caller) = \begin{cases}
\text{P1} & \text{if } caller = 1, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if } caller = 1, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if } caller = 2, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if } caller = 2, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o = \text{Active}
\end{cases}$

Where:

- $\text{COUNT}(H, v)$ returns the number of dice in hand $H$ showing value $v$
- $D_{\text{last}}$ is the most recent bid
- If caller wins: bid was invalid (caller was correct to challenge)
- If bidder wins: bid was valid (caller was wrong to challenge)

$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$ - baseline prior over opponent's hand

- Any bid given from opponent is not valid proof of the hand state.

HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $b(H_2 \mid H_1, D_last)$

$D_last = (v*, a*)$ - value, amount

$P_{valid}(D_{last} \mid H_1) = P(COUNT(H_1 \cup H_2, v*) \ge a* | H_1)$

$c_1 = COUNT(H_1, v*)$, $c_2 = COUNT(H_2, v*)$ - number of dice in each hand with value $v*$
$k = a* - c_1$ - additional dice needed from opponent for bid to be valid

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k)$

Because we don't know opponent's hand / policy, $c_2 \sim Binomial(N, p = \frac{1}{6})$

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$

- If $k > N$, then $P_{valid} = 0$ - impossible bid
- If $k \le 0$, then $P_{valid} = 1$ - our hand satisfies bid

### Notes:

- Modeled from Player 1 perspective. Opponent actions handled by system internally.
```

Updating the code to match MATH.md. Fixing the bidding rules:

```python
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
    - Each subsequent bid must increase either quantity OR face value (or both)
    - You can increase quantity while keeping same face value
    - You can increase face value while keeping same or higher quantity
    - You can increase both quantity and face value
    - You CANNOT decrease quantity when increasing face value
    - You CANNOT decrease face value
    - You CANNOT decrease quantity when keeping same face value

    Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "four 4s" ✓ valid (increase both)
    - "three 3s" → "two 4s" ✗ invalid (increase value but decrease quantity)
    - "four 4s" → "three 3s" ✗ invalid (decrease both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current and Amount_new >= Amount_current
        True  if DiceValue_new = DiceValue_current and Amount_new > Amount_current
        False otherwise
    }
    """
    if D_last is None:
        # D = ∅: first bid - any valid bid is allowed
        return True

    # DiceValue_new > DiceValue_current and Amount_new >= Amount_current
    # When increasing value, amount must be same or higher (cannot decrease)
    if new_bid.value > D_last.value and new_bid.amount >= D_last.amount:
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
    WIN(s): Determine the winner after a LIAR call.

    WIN(s) = {
        P1  if a = C, G_o' = GameOver, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        P2  if a = C, G_o' = GameOver, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P1  if opponent calls liar, COUNT(H1 ∪ H2, D_last.Value) >= D_last.Amount
        P2  if opponent calls liar, COUNT(H1 ∪ H2, D_last.Value) < D_last.Amount
        None if G_o' = Active
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

```

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.rules import check_win_condition, count_total_dice, is_legal_bid


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
        for amount in range(last_bid.amount, max_amount + 1):  # Start from current amount
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
                self._enumerate_states(obs, states_by_depth, max_depth=30, visited=all_visited)

        print(f"Total states enumerated: {sum(len(states) for states in states_by_depth.values())}")

        # Process states in reverse depth order (backward induction)
        max_depth = max(states_by_depth.keys()) if states_by_depth else 0
        print(f"Computing optimal policy using backward induction (max depth: {max_depth})...")

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
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

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
                        self._enumerate_states(new_obs, states_by_depth, max_depth, visited)

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

```

```python
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

        Priority order: increase value first, then increase amount.
        """
        # Try increasing value with minimum amount (1)
        for value in range(last_bid.value + 1, 7):
            bid = Bid(value=value, amount=1)
            if is_legal_bid(bid, last_bid):
                return bid

        # Then try same value with increased amount
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)

```

```python
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

```

Verifying win condition documentation matches MATH.md:

Updating the win condition documentation to match MATH.md:

```python
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
    - Each subsequent bid must increase either quantity OR face value (or both)
    - You can increase quantity while keeping same face value
    - You can increase face value while keeping same or higher quantity
    - You can increase both quantity and face value
    - You CANNOT decrease quantity when increasing face value
    - You CANNOT decrease face value
    - You CANNOT decrease quantity when keeping same face value

    Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "four 4s" ✓ valid (increase both)
    - "three 3s" → "two 4s" ✗ invalid (increase value but decrease quantity)
    - "four 4s" → "three 3s" ✗ invalid (decrease both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current and Amount_new >= Amount_current
        True  if DiceValue_new = DiceValue_current and Amount_new > Amount_current
        False otherwise
    }
    """
    if D_last is None:
        # D = ∅: first bid - any valid bid is allowed
        return True

    # DiceValue_new > DiceValue_current and Amount_new >= Amount_current
    # When increasing value, amount must be same or higher (cannot decrease)
    if new_bid.value > D_last.value and new_bid.amount >= D_last.amount:
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

```

## Summary

### 1. Fixed MATH.md to match standard Liar's Dice rules

- Bidding rules: when increasing face value, amount must be >= current amount (not any amount >= 1)
  - "three 3s" → "two 4s" is now INVALID (was incorrectly marked valid)
  - "three 3s" → "three 4s" is VALID (increase value, same quantity)
  - "three 3s" → "four 4s" is VALID (increase both)
- Clarified win conditions: documented WIN(s, caller) with all cases
- Fixed transition probabilities: clarified state transitions
- Updated LEGAL(D) formula: added `Amount_new >= Amount_current` when increasing value

### 2. Updated code to match MATH.md

- `src/rules.py`:
  - Updated `is_legal_bid()` to require `amount >= current_amount` when increasing value
  - Updated win condition documentation to match MATH.md

- `src/dp_solver.py`:
  - Updated `generate_all_legal_bids()` to start amount from `last_bid.amount` (not 1) when increasing value

- `src/opponent_policies.py`:
  - Updated `_find_lowest_truthful_bid()` to respect the new rule
  - Updated `_find_lowest_legal_bid()` to use `last_bid.amount` (not 1) when increasing value

All code now matches MATH.md, which reflects standard Liar's Dice rules. The key change: when increasing face value, you cannot decrease the quantity—it must stay the same or increase.

---

**User**

we are using the sum of weighted reward or something when comuting the policy in the solver right?

---

**Cursor**

Checking how rewards and values are computed in the DP solver:

Checking how `hand_prob` is initialized and used:

Yes. The solver computes expected values by summing weighted rewards over all possible opponent hands.

Here's how it works:

1. **Uniform prior over opponent's hand** (line 100):

```python
self.hand_prob = 1.0 / self.num_hands  # Uniform prior
```

2. **When calling LIAR** (`_compute_liar_value`, line 277):

```python
reward = compute_reward(state, caller=1, last_bid=last_bid)
total_value += reward * self.hand_prob  # Weighted sum
```

3. **When making a bid** (`_compute_bid_value`):
   - If opponent calls LIAR (line 302): `total_value += reward * self.hand_prob`
   - If opponent makes a bid (line 315): `total_value += value * self.hand_prob`

This computes the expected value:
$$E[V] = \sum_{H_2} P(H_2) \cdot \text{reward/value}(H_2) = \sum_{H_2} \frac{1}{|\text{all hands}|} \cdot \text{reward/value}(H_2)$$

Where:

- Each opponent hand $H_2$ has uniform probability $P(H_2) = \frac{1}{6^N}$
- We sum over all $6^N$ possible opponent hands
- The result is the expected value from Player 1's perspective

This matches the MATH.md formulation where $b(H_2) = \text{Uniform}(\{1,2,3,4,5,6\}^N)$.

The implementation is correct.

```python
self.hand_prob = 1.0 / self.num_hands  # Uniform prior
```

```python
reward = compute_reward(state, caller=1, last_bid=last_bid)
total_value += reward * self.hand_prob  # Weighted sum
```

---

**User**

so theres -1 for us losing and +1 for winning and 0 otherwise right

---

**Cursor**

Checking the reward function to confirm:

Yes. The reward structure is:

- +1.0 if Player 1 wins
- -1.0 if Player 2 wins (Player 1 loses)
- 0.0 otherwise (non-terminal states)

This matches the code:

```70:83:src/dp_solver.py
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
```

And MATH.md:

$$
R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}
$$

Since this is from Player 1's perspective:

- +1 when Player 1 wins
- -1 when Player 2 wins (Player 1 loses)
- 0 for non-terminal states

The DP solver maximizes expected reward, so it chooses actions that maximize the probability-weighted sum of these rewards.

```python
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
```

```python
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
```

---

**User**

check to see if the implementation is good and working and functional again

---

**Cursor**

Reviewing the implementation for correctness and functionality:

Checking consistency and potential issues:

Verifying consistency and testing edge cases:

Checking Python version and testing with a simpler approach:

---

**User**

@AGENTS.md

---

**Cursor**

Using `uv` to test the implementation:

Testing additional critical components:

Verifying imports and checking for any missing dependencies:

## Implementation status: functional

### Tests passed

1. Bid generation: legal bids generated correctly; rejects invalid bids (e.g., decreasing amount when increasing value)
2. State keys: `state_key` functions match between `dp_solver.py` and `policy.py`
3. Win conditions: correctly determine winner based on bid validity
4. Rewards: +1.0 for P1 win, -1.0 for P2 win, 0.0 otherwise
5. Opponent policy: calls LIAR on impossible bids; makes truthful/lowest legal bids
6. Imports: all modules import successfully

### Code quality

- No linter errors
- No TODO/FIXME comments
- Consistent with MATH.md rules
- Properly marginalizes over opponent's hand using uniform prior
- Backward induction implemented correctly

### Verified components

- `is_legal_bid()`: enforces standard Liar's Dice rules
- `generate_all_legal_bids()`: generates bids in priority order
- `check_win_condition()`: matches MATH.md win conditions
- `compute_reward()`: returns correct rewards
- `DPSolver`: computes expected values via weighted sum over opponent hands
- `TruthTellingPolicy`: implements lowest truthful legal bid strategy

### Ready to use

The implementation is functional and ready for:

- Generating policies with `generate_policy.py`
- Running simulations with `main.py`
- The DP solver correctly computes optimal actions using backward induction

All core functionality is working as expected.

---

**User**

what happens when theres no policy generated for the player 1 to use

---

**Cursor**

Checking how the policy handles a missing policy file:

When there's no policy file, the system falls back to a heuristic policy:

1. **Initialization** (lines 38-44 in `policy.py`):
   - Prints a warning: `"Warning: Policy file {policy_file} not found. Using heuristic fallback."`
   - Sets `self.policy = {}` (empty dict)

2. **Action selection** (lines 60-75):
   - Tries to look up the state in the precomputed policy
   - If not found (always when policy is empty), calls `_heuristic_action()`

3. **Heuristic fallback** (lines 77-112):
   - **First bid**: Makes a conservative bid based on the most common die value in hand
   - **Subsequent bids**:
     - Calculates `p_valid` (probability the last bid is valid)
     - If `p_valid < 0.1`: calls LIAR
     - Otherwise: makes a higher bid (increases value or amount)

**Example flow:**

```
No policy file → Warning printed → Empty policy dict →
get_action() called → Key not found → _heuristic_action() →
Uses probability-based heuristic strategy
```

The heuristic is functional but not optimal. For optimal play, run `generate_policy.py` first to create the DP-computed policy file.

The system continues to work without a policy file, using the heuristic as a fallback.

```plaintext
No policy file → Warning printed → Empty policy dict →
get_action() called → Key not found → _heuristic_action() →
Uses probability-based heuristic strategy
```

---

**User**

it seems like the generate_policy function is glitching out

---

**Cursor**

Checking `generate_policy.py` for issues:

Checking the DP solver for issues that could cause it to hang or fail:

Testing with a smaller case to identify the issue:

The issue: with 5 dice, the state enumeration explodes. For each legal bid, it iterates over all 7776 opponent hands, creating many states. Checking the enumeration logic and adding progress tracking:

The enumeration is causing a combinatorial explosion. For 5 dice:

- 7776 possible hands
- ~120 legal bids from the first state
- For each bid, checking all 7776 opponent hands
- This creates millions of states

The enumeration approach doesn't scale. Adding progress tracking and optimizing the enumeration:

```python
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

        # Limit enumeration to prevent explosion - only enumerate unique bid histories
        # We'll compute values on-demand during backward induction
        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 1000 == 0:
                print(f"  Processed {hand_count}/{self.num_hands} hands, {len(all_visited)} states so far...")

            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            key = state_key(obs)
            if key not in all_visited:
                states_by_depth[0].append(obs)
                all_visited.add(key)
                # Recursively enumerate all reachable states from this hand
                # Limit depth more aggressively to prevent explosion
                self._enumerate_states(
                    obs, states_by_depth, max_depth=15, visited=all_visited
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

```

```python
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

        # Limit enumeration to prevent explosion - only enumerate unique bid histories
        # We'll compute values on-demand during backward induction
        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 1000 == 0:
                print(f"  Processed {hand_count}/{self.num_hands} hands, {len(all_visited)} states so far...")

            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            key = state_key(obs)
            if key not in all_visited:
                states_by_depth[0].append(obs)
                all_visited.add(key)
                # Recursively enumerate all reachable states from this hand
                # Limit depth more aggressively to prevent explosion
                self._enumerate_states(
                    obs, states_by_depth, max_depth=15, visited=all_visited
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
        # Optimize: only enumerate unique bid histories, not all opponent hands
        # We'll marginalize over opponent hands during value computation
        seen_next_states = set()
        for bid in legal_bids:
            new_bid_history = obs.bid_history + [bid]

            # Sample opponent responses to find unique reachable states
            # Instead of checking all 7776 hands, sample strategically
            # Check a subset of opponent hands to find unique bid responses
            opponent_responses = set()
            samples_to_check = min(100, self.num_hands)  # Limit samples to prevent explosion

            # Sample evenly across opponent hands
            step = max(1, self.num_hands // samples_to_check)
            for i in range(0, self.num_hands, step):
                hand2_tuple = self.all_hands[i]
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal outcome, not a new state
                    continue
                else:
                    # Track unique opponent responses (bid histories)
                    opponent_bid_history = new_bid_history + [opponent_action]
                    bid_history_key = "|".join(f"{b.value},{b.amount}" for b in opponent_bid_history)
                    opponent_responses.add(bid_history_key)

            # Create states for each unique bid history
            for bid_history_key in opponent_responses:
                # Parse bid history
                bid_parts = bid_history_key.split("|")
                opponent_bid_history = []
                for part in bid_parts:
                    v, a = map(int, part.split(","))
                    opponent_bid_history.append(Bid(v, a))

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

```

Fixing the optimization: the previous approach was incorrect. Switching to a more efficient enumeration:

```python
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

        # Limit enumeration to prevent explosion - only enumerate unique bid histories
        # We'll compute values on-demand during backward induction
        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 1000 == 0:
                print(f"  Processed {hand_count}/{self.num_hands} hands, {len(all_visited)} states so far...")

            # Start with empty bid history (depth 0)
            obs = ObservableState(
                hand1=hand1, bid_history=[], game_status=GameStatus.ACTIVE
            )
            key = state_key(obs)
            if key not in all_visited:
                states_by_depth[0].append(obs)
                all_visited.add(key)
                # Recursively enumerate all reachable states from this hand
                # Limit depth more aggressively to prevent explosion
                self._enumerate_states(
                    obs, states_by_depth, max_depth=15, visited=all_visited
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

        # For each legal bid, find unique opponent responses
        # Key insight: we only need to enumerate unique bid histories, not all opponent hands
        # The opponent policy is deterministic given their hand, so we can find all unique responses
        seen_next_states = set()
        for bid in legal_bids:
            new_bid_history = obs.bid_history + [bid]

            # Find all unique opponent bid responses (not all opponent hands)
            # We'll check opponent hands but only track unique bid histories
            unique_opponent_bids = set()

            # Limit check to prevent explosion - check every Nth hand
            # This finds unique responses without checking all 7776 hands
            check_step = max(1, self.num_hands // 200)  # Check ~200 hands max per bid

            for i in range(0, self.num_hands, check_step):
                hand2_tuple = self.all_hands[i]
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal, not a new state
                    continue
                else:
                    # Track unique opponent bid (value, amount) pairs
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

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

```

Testing the optimized version:

Optimized the enumeration. Summary:

## Issue

The DP solver was enumerating too many states:

- For each legal bid (~120 from first state)
- For each opponent hand (7776 for 5 dice)
- This creates millions of states and causes a combinatorial explosion

## Fixes Applied

1. Added progress tracking: shows progress every 1000 hands
2. Reduced max depth: from 30 to 15 to limit recursion
3. Optimized enumeration: samples ~200 opponent hands per bid instead of all 7776, finding unique opponent responses without checking every hand

## Current Status

- Works for 1-2 dice (tested)
- Should be faster for 5 dice, but may still take time due to state space size
- The enumeration now samples strategically instead of checking all hands

The policy generation should work now, but for 5 dice it may take several minutes to hours depending on the state space. The optimization reduces the work from O(bids × 7776) to O(bids × 200) per level, which should help significantly.

Try running `generate_policy.py` again; it should show progress and complete, though it may take a while for 5 dice.

---

**User**

na it should check all the bids instead of sampling thats what i want, however i want actual tracking instead of it being stuck on processing x number of states

---

**Cursor**

Reverting sampling to check all opponent hands and adding detailed progress tracking:

```python
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

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far...")

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

        # For each legal bid, find unique opponent responses
        # Key insight: we only need to enumerate unique bid histories, not all opponent hands
        # The opponent policy is deterministic given their hand, so we can find all unique responses
        seen_next_states = set()
        for bid in legal_bids:
            new_bid_history = obs.bid_history + [bid]

            # Find all unique opponent bid responses (not all opponent hands)
            # We'll check opponent hands but only track unique bid histories
            unique_opponent_bids = set()

            # Limit check to prevent explosion - check every Nth hand
            # This finds unique responses without checking all 7776 hands
            check_step = max(1, self.num_hands // 200)  # Check ~200 hands max per bid

            for i in range(0, self.num_hands, check_step):
                hand2_tuple = self.all_hands[i]
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal, not a new state
                    continue
                else:
                    # Track unique opponent bid (value, amount) pairs
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

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

```

```python
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

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far...")

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
        # Check ALL opponent hands to find all unique reachable states
        seen_next_states = set()
        total_bids = len(legal_bids)

        for bid_idx, bid in enumerate(legal_bids):
            new_bid_history = obs.bid_history + [bid]

            # Check all opponent hands to find unique responses
            unique_opponent_bids = set()
            for hand2_tuple in self.all_hands:
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal outcome, not a new state
                    continue
                else:
                    # Track unique opponent bid (value, amount) pairs
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

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

```

```python
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

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far...")

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

            for state_idx, obs in enumerate(states):
                if state_idx % 1000 == 0 and state_idx > 0:
                    print(f"    Depth {depth}: {state_idx}/{len(states)} states computed...")
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
        # Check ALL opponent hands to find all unique reachable states
        seen_next_states = set()
        total_bids = len(legal_bids)

        for bid_idx, bid in enumerate(legal_bids):
            new_bid_history = obs.bid_history + [bid]

            # Check all opponent hands to find unique responses
            unique_opponent_bids = set()
            for hand2_tuple in self.all_hands:
                hand2 = list(hand2_tuple)
                opponent_action = self.opponent_policy.get_action(
                    hand2, new_bid_history
                )

                if opponent_action is None:
                    # Opponent calls LIAR - terminal outcome, not a new state
                    continue
                else:
                    # Track unique opponent bid (value, amount) pairs
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

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

```

Adding progress tracking during enumeration when checking opponent hands:

```python
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

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far...")

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

            for state_idx, obs in enumerate(states):
                if state_idx % 1000 == 0 and state_idx > 0:
                    print(f"    Depth {depth}: {state_idx}/{len(states)} states computed...")
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
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

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
        # Check ALL opponent hands to find all unique reachable states
        seen_next_states = set()
        total_bids = len(legal_bids)

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
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

                # Progress tracking: print every 10000 opponent hand checks
                enumeration_counter["calls"] += 1
                if enumeration_counter["calls"] - enumeration_counter["last_print"] >= 10000:
                    total_states = sum(len(states) for states in states_by_depth.values())
                    print(f"    Enumerating: {enumeration_counter['calls']} opponent checks, {total_states} states found...")
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
                        new_obs, states_by_depth, max_depth, visited, enumeration_counter
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

```

```python
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

        hand_count = 0
        for hand1_tuple in self.all_hands:
            hand1 = list(hand1_tuple)
            hand_count += 1
            if hand_count % 100 == 0:
                total_states = sum(len(states) for states in states_by_depth.values())
                print(f"  Hand {hand_count}/{self.num_hands}: {total_states} states enumerated so far...")

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
                    obs, states_by_depth, max_depth=30, visited=all_visited, enumeration_counter=enumeration_counter
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
                    print(f"    Depth {depth}: {state_idx}/{len(states)} states computed...")
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
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

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
        # Check ALL opponent hands to find all unique reachable states
        seen_next_states = set()
        total_bids = len(legal_bids)

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
                    unique_opponent_bids.add((opponent_action.value, opponent_action.amount))

                # Progress tracking: print every 10000 opponent hand checks
                enumeration_counter["calls"] += 1
                if enumeration_counter["calls"] - enumeration_counter["last_print"] >= 10000:
                    total_states = sum(len(states) for states in states_by_depth.values())
                    print(f"    Enumerating: {enumeration_counter['calls']} opponent checks, {total_states} states found...")
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
                        new_obs, states_by_depth, max_depth, visited, enumeration_counter
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

```

---

**User**

change the default runtime to be with 4 dice instead too

---

**Cursor**

Updating the default number of dice from 5 to 4:

Updating defaults from 5 to 4:

```python
"""Generate optimal DP policy for Liar's Dice."""

from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = 4

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(num_dice, opponent_policy)

    # Solve
    solver.solve()

    # Save policy
    policy_file = f"policy_{num_dice}dice.json"
    solver.save_policy(policy_file)

    print()
    print("=" * 60)
    print("Policy generation complete!")
    print(f"Policy saved to: {policy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Main entry point for Liar's Dice simulation."""

import os

from src.simulation import GameSimulation


def main():
    """Run the Liar's Dice simulation with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = 4
    policy_file = f"policy_{num_dice}dice.json"

    # Check if policy file exists
    if not os.path.exists(policy_file):
        print(
            f"Warning: Policy file '{policy_file}' not found.\n"
            f"Run 'python generate_policy.py' first to generate optimal policy.\n"
            f"Using heuristic fallback for now.\n"
        )

    # Create simulation with default 4 dice per player
    simulation = GameSimulation(num_dice_per_player=num_dice)

    # Run a single game
    winner, final_state = simulation.simulate_game(verbose=True)

    print("\n=== Game Over ===")
    print(f"Final winner: Player {winner}")
    print(f"Total bids made: {len(final_state.bid_history)}")


if __name__ == "__main__":
    main()

```

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(self, num_dice_per_player: int = 4):
        self.num_dice_per_player = num_dice_per_player
        self.player1_policy = OptimalPolicy(num_dice_per_player)
        self.opponent_policy = TruthTellingPolicy(num_dice_per_player)

    def roll_dice(self, num_dice: int) -> list[int]:
        """Roll N dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def simulate_game(self, verbose: bool = True) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state)."""
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Randomly choose starting player
        starting_player = random.randint(1, 2)

        if verbose:
            print("=" * 60)
            print("GAME START")
            print("=" * 60)
            print(f"Player 1 hand: {hand1}")
            print(f"Player 2 hand: {hand2}")
            print(f"Starting player: Player {starting_player}")
            print("Bid history: []")
            print("-" * 60)

        turn_number = 1  # Track turns from Player 1's perspective
        current_player = starting_player

        while state.game_status == GameStatus.ACTIVE:
            if current_player == 1:
                if verbose:
                    print(f"\n--- Turn {turn_number} ---")
                    print(f"Current bid history: {[str(b) for b in state.bid_history]}")

            if current_player == 1:
                # Player 1's turn
                obs = observable_from_state(state)

                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    prob = p_valid(last_bid, obs.hand1, self.num_dice_per_player)
                    print(f"Player 1 evaluating last bid: {last_bid}")
                    print(f"  P(valid) = {prob:.4f}")
                    print(
                        f"  Player 1 has {sum(1 for d in obs.hand1 if d == last_bid.value)} {last_bid.value}s"
                    )

                action = self.player1_policy.get_action(obs)

                if action is None:
                    # Player 1 calls LIAR
                    if verbose:
                        print("Player 1 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=1, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 1 makes a bid
                    if verbose:
                        print(f"Player 1 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 2
            else:
                # Player 2's turn
                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    # Calculate what Player 2 sees (they know their own hand)
                    c2 = sum(1 for d in state.hand2 if d == last_bid.value)
                    max_possible = c2 + self.num_dice_per_player
                    print(f"Player 2 evaluating last bid: {last_bid}")
                    print(f"  Player 2 has {c2} {last_bid.value}s")
                    print(
                        f"  Max possible total: {max_possible} (if Player 1 has all {last_bid.value}s)"
                    )
                    print(f"  Bid requires: {last_bid.amount}")

                action = self.opponent_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if verbose:
                        print("Player 2 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=2, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 2 makes a bid
                    if verbose:
                        print(f"Player 2 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 1
                    turn_number += 1  # Increment turn after Player 2 responds

        # Should not reach here
        return 0, state

```

Updated the default number of dice from 5 to 4 in:

1. **`generate_policy.py`**: `num_dice = 4`
2. **`main.py`**: `num_dice = 4` (and updated comment)
3. **`src/simulation.py`**: default parameter `num_dice_per_player: int = 4`

Running `generate_policy.py` or `main.py` will use 4 dice by default. This should be faster than 5 dice (6^4 = 1296 possible hands vs 6^5 = 7776).

---

**User**

what does compute optimal policy using backward induction mean

---

**Cursor**

Backward induction solves the game by starting at terminal states and working backwards.

## How it works in this solver

1. Terminal states (game over): known values
   - If Player 1 wins: +1.0
   - If Player 2 wins: -1.0

2. Process states in reverse depth order:
   - Start with the deepest bid histories (e.g., depth 10)
   - Then depth 9, 8, 7, ... down to depth 0 (no bids yet)

3. For each state, compute optimal action:
   - Look at all possible actions (call LIAR or make a bid)
   - For each action, compute expected value by looking at future states
   - Since we process backwards, future states already have computed values
   - Choose the action with highest expected value

## Example

```
Depth 3: [bid1, bid2, bid3] → Already computed (future states)
Depth 2: [bid1, bid2] → Can compute optimal action (knows what happens at depth 3)
Depth 1: [bid1] → Can compute optimal action (knows what happens at depth 2)
Depth 0: [] → Can compute optimal action (knows what happens at depth 1)
```

## Why backwards?

- Forward: you don't know future values yet
- Backward: future values are already computed, so you can choose the best action now

This ensures each state's value reflects the optimal strategy from that point onward, which is what we want for the optimal policy.

```plaintext
Depth 3: [bid1, bid2, bid3] → Already computed (future states)
Depth 2: [bid1, bid2] → Can compute optimal action (knows what happens at depth 3)
Depth 1: [bid1] → Can compute optimal action (knows what happens at depth 2)
Depth 0: [] → Can compute optimal action (knows what happens at depth 1)
```

---

**User**

change default to 3 now

---

**Cursor**

Updating the default number of dice from 4 to 3:

```python
"""Generate optimal DP policy for Liar's Dice."""

from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = 3

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(num_dice, opponent_policy)

    # Solve
    solver.solve()

    # Save policy
    policy_file = f"policy_{num_dice}dice.json"
    solver.save_policy(policy_file)

    print()
    print("=" * 60)
    print("Policy generation complete!")
    print(f"Policy saved to: {policy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Main entry point for Liar's Dice simulation."""

import os

from src.simulation import GameSimulation


def main():
    """Run the Liar's Dice simulation with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = 3
    policy_file = f"policy_{num_dice}dice.json"

    # Check if policy file exists
    if not os.path.exists(policy_file):
        print(
            f"Warning: Policy file '{policy_file}' not found.\n"
            f"Run 'python generate_policy.py' first to generate optimal policy.\n"
            f"Using heuristic fallback for now.\n"
        )

    # Create simulation with default 3 dice per player
    simulation = GameSimulation(num_dice_per_player=num_dice)

    # Run a single game
    winner, final_state = simulation.simulate_game(verbose=True)

    print("\n=== Game Over ===")
    print(f"Final winner: Player {winner}")
    print(f"Total bids made: {len(final_state.bid_history)}")


if __name__ == "__main__":
    main()

```

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(self, num_dice_per_player: int = 3):
        self.num_dice_per_player = num_dice_per_player
        self.player1_policy = OptimalPolicy(num_dice_per_player)
        self.opponent_policy = TruthTellingPolicy(num_dice_per_player)

    def roll_dice(self, num_dice: int) -> list[int]:
        """Roll N dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def simulate_game(self, verbose: bool = True) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state)."""
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Randomly choose starting player
        starting_player = random.randint(1, 2)

        if verbose:
            print("=" * 60)
            print("GAME START")
            print("=" * 60)
            print(f"Player 1 hand: {hand1}")
            print(f"Player 2 hand: {hand2}")
            print(f"Starting player: Player {starting_player}")
            print("Bid history: []")
            print("-" * 60)

        turn_number = 1  # Track turns from Player 1's perspective
        current_player = starting_player

        while state.game_status == GameStatus.ACTIVE:
            if current_player == 1:
                if verbose:
                    print(f"\n--- Turn {turn_number} ---")
                    print(f"Current bid history: {[str(b) for b in state.bid_history]}")

            if current_player == 1:
                # Player 1's turn
                obs = observable_from_state(state)

                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    prob = p_valid(last_bid, obs.hand1, self.num_dice_per_player)
                    print(f"Player 1 evaluating last bid: {last_bid}")
                    print(f"  P(valid) = {prob:.4f}")
                    print(
                        f"  Player 1 has {sum(1 for d in obs.hand1 if d == last_bid.value)} {last_bid.value}s"
                    )

                action = self.player1_policy.get_action(obs)

                if action is None:
                    # Player 1 calls LIAR
                    if verbose:
                        print("Player 1 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=1, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 1 makes a bid
                    if verbose:
                        print(f"Player 1 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 2
            else:
                # Player 2's turn
                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    # Calculate what Player 2 sees (they know their own hand)
                    c2 = sum(1 for d in state.hand2 if d == last_bid.value)
                    max_possible = c2 + self.num_dice_per_player
                    print(f"Player 2 evaluating last bid: {last_bid}")
                    print(f"  Player 2 has {c2} {last_bid.value}s")
                    print(
                        f"  Max possible total: {max_possible} (if Player 1 has all {last_bid.value}s)"
                    )
                    print(f"  Bid requires: {last_bid.amount}")

                action = self.opponent_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if verbose:
                        print("Player 2 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=2, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 2 makes a bid
                    if verbose:
                        print(f"Player 2 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 1
                    turn_number += 1  # Increment turn after Player 2 responds

        # Should not reach here
        return 0, state

```

Updated the default number of dice from 4 to 3 in:

1. **`generate_policy.py`**: `num_dice = 3`
2. **`main.py`**: `num_dice = 3` (and updated comment)
3. **`src/simulation.py`**: default parameter `num_dice_per_player: int = 3`

With 3 dice, there are 6^3 = 216 possible hands per player, which should make policy generation faster.

---

**User**

change to 2

---

**Cursor**

Updating the default number of dice from 3 to 2:

```python
"""Generate optimal DP policy for Liar's Dice."""

from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = 2

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(num_dice, opponent_policy)

    # Solve
    solver.solve()

    # Save policy
    policy_file = f"policy_{num_dice}dice.json"
    solver.save_policy(policy_file)

    print()
    print("=" * 60)
    print("Policy generation complete!")
    print(f"Policy saved to: {policy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Main entry point for Liar's Dice simulation."""

import os

from src.simulation import GameSimulation


def main():
    """Run the Liar's Dice simulation with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = 2
    policy_file = f"policy_{num_dice}dice.json"

    # Check if policy file exists
    if not os.path.exists(policy_file):
        print(
            f"Warning: Policy file '{policy_file}' not found.\n"
            f"Run 'python generate_policy.py' first to generate optimal policy.\n"
            f"Using heuristic fallback for now.\n"
        )

    # Create simulation with default 2 dice per player
    simulation = GameSimulation(num_dice_per_player=num_dice)

    # Run a single game
    winner, final_state = simulation.simulate_game(verbose=True)

    print("\n=== Game Over ===")
    print(f"Final winner: Player {winner}")
    print(f"Total bids made: {len(final_state.bid_history)}")


if __name__ == "__main__":
    main()

```

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(self, num_dice_per_player: int = 2):
        self.num_dice_per_player = num_dice_per_player
        self.player1_policy = OptimalPolicy(num_dice_per_player)
        self.opponent_policy = TruthTellingPolicy(num_dice_per_player)

    def roll_dice(self, num_dice: int) -> list[int]:
        """Roll N dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def simulate_game(self, verbose: bool = True) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state)."""
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Randomly choose starting player
        starting_player = random.randint(1, 2)

        if verbose:
            print("=" * 60)
            print("GAME START")
            print("=" * 60)
            print(f"Player 1 hand: {hand1}")
            print(f"Player 2 hand: {hand2}")
            print(f"Starting player: Player {starting_player}")
            print("Bid history: []")
            print("-" * 60)

        turn_number = 1  # Track turns from Player 1's perspective
        current_player = starting_player

        while state.game_status == GameStatus.ACTIVE:
            if current_player == 1:
                if verbose:
                    print(f"\n--- Turn {turn_number} ---")
                    print(f"Current bid history: {[str(b) for b in state.bid_history]}")

            if current_player == 1:
                # Player 1's turn
                obs = observable_from_state(state)

                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    prob = p_valid(last_bid, obs.hand1, self.num_dice_per_player)
                    print(f"Player 1 evaluating last bid: {last_bid}")
                    print(f"  P(valid) = {prob:.4f}")
                    print(
                        f"  Player 1 has {sum(1 for d in obs.hand1 if d == last_bid.value)} {last_bid.value}s"
                    )

                action = self.player1_policy.get_action(obs)

                if action is None:
                    # Player 1 calls LIAR
                    if verbose:
                        print("Player 1 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=1, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 1 makes a bid
                    if verbose:
                        print(f"Player 1 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 2
            else:
                # Player 2's turn
                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    # Calculate what Player 2 sees (they know their own hand)
                    c2 = sum(1 for d in state.hand2 if d == last_bid.value)
                    max_possible = c2 + self.num_dice_per_player
                    print(f"Player 2 evaluating last bid: {last_bid}")
                    print(f"  Player 2 has {c2} {last_bid.value}s")
                    print(
                        f"  Max possible total: {max_possible} (if Player 1 has all {last_bid.value}s)"
                    )
                    print(f"  Bid requires: {last_bid.amount}")

                action = self.opponent_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if verbose:
                        print("Player 2 calls LIAR!")
                    state.game_status = GameStatus.GAME_OVER
                    winner = check_win_condition(
                        state, caller=2, D_last=state.bid_history[-1]
                    )
                    if verbose:
                        last_bid = state.bid_history[-1]
                        total = sum(
                            1 for d in state.hand1 + state.hand2 if d == last_bid.value
                        )
                        print("\n" + "=" * 60)
                        print("GAME OVER")
                        print("=" * 60)
                        print(f"Last bid: {last_bid}")
                        print(f"Total {last_bid.value}s across both hands: {total}")
                        print(f"Player 1 hand: {state.hand1}")
                        print(f"Player 2 hand: {state.hand2}")
                        print(f"Winner: Player {winner}")
                        print("=" * 60)
                    return winner, state
                else:
                    # Player 2 makes a bid
                    if verbose:
                        print(f"Player 2 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 1
                    turn_number += 1  # Increment turn after Player 2 responds

        # Should not reach here
        return 0, state

```

Updated the default number of dice from 3 to 2 in:

1. **`generate_policy.py`**: `num_dice = 2`
2. **`main.py`**: `num_dice = 2` (and updated comment)
3. **`src/simulation.py`**: default parameter `num_dice_per_player: int = 2`

With 2 dice, there are 6^2 = 36 possible hands per player, which should make policy generation much faster.

---

**User**

how long do you estimate it will take cause im trying with 2 now and its taking a while

---

**Cursor**

Estimating runtime and checking the current state:

Estimating runtime: with 2 dice, `max_amount=20` creates ~120 first bids. Checking all 36 opponent hands per bid leads to many operations. Providing a more accurate estimate:

## Time estimate for 2 dice: 10–60 minutes

**Complexity:**

- 36 hands per player
- ~120 first bid options (6 values × 20 amounts)
- 36 opponent hands to check per bid
- First level: 36 × 120 × 36 = 155,520 operations
- With recursion up to depth 30, this can create hundreds of thousands of states

**What to watch for:**

- Progress updates every 10,000 opponent checks
- "Hand X/36" updates every 100 hands
- "Depth X: Y states" during backward induction

**To speed it up (if needed):**
Reduce `max_amount` in `generate_all_legal_bids()` from 20 to 10 or 12. This cuts first bids from 120 to 60 or 72, roughly halving the work.

The enumeration phase is the bottleneck. Once it finishes, backward induction is usually faster. If you see progress updates, it’s working; it may just take a while.

---

**User**

how is recursion depth used? like what and where is it used for

---

**Cursor**

Checking how recursion depth is used in the DP solver:

Explaining how recursion depth works:

## How recursion depth works

**Depth = length of bid history** (number of bids made so far)

### Where it's used:

1. **In `_enumerate_states()` (line 178-180)**:

```python
depth = len(obs.bid_history)  # Count how many bids have been made
if depth > max_depth:  # If too many bids, stop recursing
    return
```

2. **What it prevents:**
   - Without `max_depth`, enumeration could continue indefinitely
   - Each bid adds to the history, creating new states
   - `max_depth=30` means: stop if bid history has more than 30 bids

3. **Example:**

```
Depth 0: [] (no bids yet)
Depth 1: [bid1] (one bid made)
Depth 2: [bid1, bid2] (two bids)
Depth 3: [bid1, bid2, bid3] (three bids)
...
Depth 30: [bid1, bid2, ..., bid30] (30 bids - MAX)
Depth 31: STOP - won't enumerate further
```

4. **Why it matters:**
   - Limits the state space explosion
   - Most games end before 30 bids (someone calls LIAR)
   - Prevents infinite recursion if the game somehow continues

**In practice:** If a game reaches 30 bids without anyone calling LIAR, enumeration stops at that point. The solver still computes optimal actions for all states up to depth 30, which covers most realistic games.

```python
depth = len(obs.bid_history)  # Count how many bids have been made
if depth > max_depth:  # If too many bids, stop recursing
    return
```

```plaintext
Depth 0: [] (no bids yet)
Depth 1: [bid1] (one bid made)
Depth 2: [bid1, bid2] (two bids)
Depth 3: [bid1, bid2, bid3] (three bids)
...
Depth 30: [bid1, bid2, ..., bid30] (30 bids - MAX)
Depth 31: STOP - won't enumerate further
```

---

**User**

but the thing is theres only x number of legal bids per state, so the max depth should just be number of possible bids, then the current depth instead of increasing by 1 per iteration it should increase by the bids position relative to the first bid, etc.

---

**Cursor**

Updating depth tracking to use bid position in the total ordering instead of a simple counter:

Updating depth tracking to use bid position in the total ordering instead of bid count:

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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
                    max_depth=30,
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
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

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

```

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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
                    max_depth=30,
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
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

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

```

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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
        """Recursively enumerate all reachable states from a given state."""
        if visited is None:
            visited = set()
        if enumeration_counter is None:
            enumeration_counter = {"calls": 0, "last_print": 0}

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

```

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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
        legal_bids = generate_all_legal_bids(last_bid)

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

```

Updating calls to `generate_all_legal_bids` to use the solver's max_value and max_amount:

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

```python
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

Finding and updating the remaining call:

Verifying the implementation:

Updated depth tracking to use bid position in the total ordering.

## Changes Made:

1. **Added `get_bid_position()`**: Returns the position of a bid in the total ordering
   - (1,1) = position 0
   - (1,2) = position 1
   - (2,1) = position 20
   - (6,20) = position 119

2. **Added `get_max_bid_position()`**: Returns total number of possible bids (120 for max_value=6, max_amount=20)

3. **Updated depth calculation**:
   - **Before**: `depth = len(obs.bid_history)` (just count of bids)
   - **After**: `depth = get_bid_position(last_bid)` (position in bid ordering)

4. **Updated max_depth**:
   - **Before**: `max_depth = 30` (arbitrary limit)
   - **After**: `max_depth = get_max_bid_position()` (120 = total possible bids)

## Benefits:

- **More accurate**: Depth reflects progress through the bid space, not just bid count
- **Better bound**: max_depth is the actual maximum (120 bids), not an arbitrary 30
- **More efficient**: Stops when reaching the highest possible bid position

Depth now increases by the bid's position relative to the start, not just +1 per bid. This should improve enumeration efficiency.

---

**User**

extract out the parameters to a parameters.py file in the root, that all other stuff use

this should just include constant variables that are used for system parameters, like number of dice, which is then in-turn used for stuff like bid space

---

**Cursor**

Creating `parameters.py` and updating files to use it:

Creating `parameters.py` and updating files to use it:

```python
"""System parameters for Liar's Dice simulation.

All system-wide constants are defined here and imported by other modules.
"""

# Dice configuration
NUM_DICE_PER_PLAYER = 2

# Bid space configuration
MAX_DICE_VALUE = 6  # Maximum face value on a die (1-6)
MAX_BID_AMOUNT = 20  # Maximum amount in a bid

# Derived constants
MAX_BID_POSITION = MAX_DICE_VALUE * MAX_BID_AMOUNT  # Total number of possible bids

```

```python
"""Generate optimal DP policy for Liar's Dice."""

import os

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, NUM_DICE_PER_PLAYER
from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = NUM_DICE_PER_PLAYER

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(num_dice, opponent_policy)

    # Solve
    solver.solve()

    # Create policy directory if it doesn't exist
    policy_dir = "policy"
    os.makedirs(policy_dir, exist_ok=True)

    # Save policy
    policy_file = os.path.join(policy_dir, f"policy_{num_dice}dice.json")
    solver.save_policy(policy_file)

    print()
    print("=" * 60)
    print("Policy generation complete!")
    print(f"Policy saved to: {policy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Generate optimal DP policy for Liar's Dice."""

import os

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, NUM_DICE_PER_PLAYER
from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = NUM_DICE_PER_PLAYER

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(
        num_dice, opponent_policy, max_value=MAX_DICE_VALUE, max_amount=MAX_BID_AMOUNT
    )

    # Solve
    solver.solve()

    # Create policy directory if it doesn't exist
    policy_dir = "policy"
    os.makedirs(policy_dir, exist_ok=True)

    # Save policy
    policy_file = os.path.join(policy_dir, f"policy_{num_dice}dice.json")
    solver.save_policy(policy_file)

    print()
    print("=" * 60)
    print("Policy generation complete!")
    print(f"Policy saved to: {policy_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER
from src.simulation import GameSimulation


def main():
    """Run 100 simulations with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = NUM_DICE_PER_PLAYER
    policy_file = os.path.join("policy", f"policy_{num_dice}dice.json")

    # Check if policy file exists
    if not os.path.exists(policy_file):
        print(
            f"Warning: Policy file '{policy_file}' not found.\n"
            f"Run 'python generate_policy.py' first to generate optimal policy.\n"
            f"Using heuristic fallback for now.\n"
        )

    num_simulations = 100
    results_truth_telling = {"player1_wins": 0, "player2_wins": 0}
    results_optimal = {"player1_wins": 0, "player2_wins": 0}

    print(f"Running {num_simulations} simulations...")
    print("=" * 60)

    # Run simulations with alternating Player 2 policies
    for i in range(num_simulations):
        # Alternate Player 2 policy: even = truth_telling, odd = optimal
        player2_policy_type = "truth_telling" if i % 2 == 0 else "optimal"

        # Create simulation
        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type=player2_policy_type,
        )

        # Run game (non-verbose for batch runs)
        winner, _ = simulation.simulate_game(verbose=False)

        # Track results
        if player2_policy_type == "truth_telling":
            if winner == 1:
                results_truth_telling["player1_wins"] += 1
            else:
                results_truth_telling["player2_wins"] += 1
        else:
            if winner == 1:
                results_optimal["player1_wins"] += 1
            else:
                results_optimal["player2_wins"] += 1

        # Progress update every 10 simulations
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_simulations // 2} with Player 2 using truth-telling policy")
    print(f"  - {num_simulations // 2} with Player 2 using optimal policy")

    print("\n--- Results vs Truth-Telling Policy ---")
    total_tt = (
        results_truth_telling["player1_wins"] + results_truth_telling["player2_wins"]
    )
    if total_tt > 0:
        p1_win_rate_tt = results_truth_telling["player1_wins"] / total_tt * 100
        print(
            f"Player 1 wins: {results_truth_telling['player1_wins']}/{total_tt} ({p1_win_rate_tt:.1f}%)"
        )
        print(
            f"Player 2 wins: {results_truth_telling['player2_wins']}/{total_tt} ({100 - p1_win_rate_tt:.1f}%)"
        )

    print("\n--- Results vs Optimal Policy ---")
    total_opt = results_optimal["player1_wins"] + results_optimal["player2_wins"]
    if total_opt > 0:
        p1_win_rate_opt = results_optimal["player1_wins"] / total_opt * 100
        print(
            f"Player 1 wins: {results_optimal['player1_wins']}/{total_opt} ({p1_win_rate_opt:.1f}%)"
        )
        print(
            f"Player 2 wins: {results_optimal['player2_wins']}/{total_opt} ({100 - p1_win_rate_opt:.1f}%)"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()

```

```python
"""Game simulation engine."""

import random

from src.game_state import (
    Bid,
    GameStatus,
    ObservableState,
    State,
    observable_from_state,
)
from parameters import NUM_DICE_PER_PLAYER
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = NUM_DICE_PER_PLAYER,
        player2_policy_type: str = "truth_telling",
    ):
        """
        Args:
            num_dice_per_player: Number of dice each player has
            player2_policy_type: "truth_telling" or "optimal" (uses same policy as Player 1)
        """
        self.num_dice_per_player = num_dice_per_player
        self.player1_policy = OptimalPolicy(num_dice_per_player)

        if player2_policy_type == "optimal":
            self.player2_policy = OptimalPolicy(num_dice_per_player)
        else:
            self.player2_policy = TruthTellingPolicy(num_dice_per_player)

    def roll_dice(self, num_dice: int) -> list[int]:
        """Roll N dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def simulate_game(self, verbose: bool = True) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state)."""
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Randomly choose starting player
        starting_player = random.randint(1, 2)

        if verbose:
            print("=" * 60)
            print("GAME START")
            print("=" * 60)
            print(f"Player 1 hand: {hand1}")
            print(f"Player 2 hand: {hand2}")
            print(f"Starting player: Player {starting_player}")
            print("Bid history: []")
            print("-" * 60)

        turn_number = 1  # Track turns from Player 1's perspective
        current_player = starting_player

        while state.game_status == GameStatus.ACTIVE:
            if current_player == 1:
                if verbose:
                    print(f"\n--- Turn {turn_number} ---")
                    print(f"Current bid history: {[str(b) for b in state.bid_history]}")

            if current_player == 1:
                # Player 1's turn
                obs = observable_from_state(state)

                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    prob = p_valid(last_bid, obs.hand1, self.num_dice_per_player)
                    print(f"Player 1 evaluating last bid: {last_bid}")
                    print(f"  P(valid) = {prob:.4f}")
                    print(
                        f"  Player 1 has {sum(1 for d in obs.hand1 if d == last_bid.value)} {last_bid.value}s"
                    )

                action = self.player1_policy.get_action(obs)

                if action is None:
                    # Player 1 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print(
                                "Player 1 cannot call LIAR with no bids, making minimal bid"
                            )
                    else:
                        if verbose:
                            print("Player 1 calls LIAR!")
                        state.game_status = GameStatus.GAME_OVER
                        winner = check_win_condition(
                            state, caller=1, D_last=state.bid_history[-1]
                        )
                        if verbose:
                            last_bid = state.bid_history[-1]
                            total = sum(
                                1
                                for d in state.hand1 + state.hand2
                                if d == last_bid.value
                            )
                            print("\n" + "=" * 60)
                            print("GAME OVER")
                            print("=" * 60)
                            print(f"Last bid: {last_bid}")
                            print(f"Total {last_bid.value}s across both hands: {total}")
                            print(f"Player 1 hand: {state.hand1}")
                            print(f"Player 2 hand: {state.hand2}")
                            print(f"Winner: Player {winner}")
                            print("=" * 60)
                        return winner, state
                else:
                    # Player 1 makes a bid
                    if verbose:
                        print(f"Player 1 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 2
            else:
                # Player 2's turn
                if verbose and state.bid_history:
                    last_bid = state.bid_history[-1]
                    # Calculate what Player 2 sees (they know their own hand)
                    c2 = sum(1 for d in state.hand2 if d == last_bid.value)
                    max_possible = c2 + self.num_dice_per_player
                    print(f"Player 2 evaluating last bid: {last_bid}")
                    print(f"  Player 2 has {c2} {last_bid.value}s")
                    print(
                        f"  Max possible total: {max_possible} (if Player 1 has all {last_bid.value}s)"
                    )
                    print(f"  Bid requires: {last_bid.amount}")

                # Player 2's action depends on policy type
                if isinstance(self.player2_policy, OptimalPolicy):
                    # Player 2 uses optimal policy (observable state from their perspective)
                    obs2 = ObservableState(
                        hand1=state.hand2,  # Player 2's hand is their "hand1"
                        bid_history=state.bid_history.copy(),
                        game_status=state.game_status,
                    )
                    action = self.player2_policy.get_action(obs2)
                else:
                    # Player 2 uses truth-telling policy
                    action = self.player2_policy.get_action(
                        state.hand2, state.bid_history
                    )

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print(
                                "Player 2 cannot call LIAR with no bids, making minimal bid"
                            )
                    else:
                        if verbose:
                            print("Player 2 calls LIAR!")
                        state.game_status = GameStatus.GAME_OVER
                        winner = check_win_condition(
                            state, caller=2, D_last=state.bid_history[-1]
                        )
                        if verbose:
                            last_bid = state.bid_history[-1]
                            total = sum(
                                1
                                for d in state.hand1 + state.hand2
                                if d == last_bid.value
                            )
                            print("\n" + "=" * 60)
                            print("GAME OVER")
                            print("=" * 60)
                            print(f"Last bid: {last_bid}")
                            print(f"Total {last_bid.value}s across both hands: {total}")
                            print(f"Player 1 hand: {state.hand1}")
                            print(f"Player 2 hand: {state.hand2}")
                            print(f"Winner: Player {winner}")
                            print("=" * 60)
                        return winner, state
                else:
                    # Player 2 makes a bid
                    if verbose:
                        print(f"Player 2 bids: {action}")
                    state.bid_history.append(action)
                    current_player = 1
                    turn_number += 1  # Increment turn after Player 2 responds

        # Should not reach here
        return 0, state

```

```python
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
    last_bid: Bid | None, max_value: int = 6, max_amount: int = 20
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

```python
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
    last_bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
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


def get_bid_position(bid: Bid | None, max_value: int = 6, max_amount: int = 20) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

```python
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
    last_bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
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


def get_bid_position(bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT) -> int:
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


def get_max_bid_position(max_value: int = 6, max_amount: int = 20) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

```python
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
    last_bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
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


def get_bid_position(bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT) -> int:
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


def get_max_bid_position(max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = 6, max_amount: int = 20):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

```python
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
    last_bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT
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


def get_bid_position(bid: Bid | None, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT) -> int:
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


def get_max_bid_position(max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT) -> int:
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

    def __init__(self, num_dice_per_player: int, opponent_policy: TruthTellingPolicy, max_value: int = MAX_DICE_VALUE, max_amount: int = MAX_BID_AMOUNT):
        self.num_dice_per_player = num_dice_per_player
        self.opponent_policy = opponent_policy
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_max_bid_position(max_value, max_amount)  # Total number of possible bids

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

```

Verifying imports and testing:

Checking for other files that need updates:

Updating remaining hardcoded values:

---

**User**

what the heck is max bid amount like what is it represent isn't it just the same as number of dice

---

**Cursor**

Checking what `MAX_BID_AMOUNT` represents and whether it should be derived from the number of dice:

Updating `MAX_BID_AMOUNT` to be derived from the number of dice. With 2 dice per player and 2 players, the maximum possible bid amount is 4 (all dice show the same value). Updating `parameters.py`:

```python
"""System parameters for Liar's Dice simulation.

All system-wide constants are defined here and imported by other modules.
"""

# Dice configuration
NUM_DICE_PER_PLAYER = 2
NUM_PLAYERS = 2  # Number of players in the game

# Bid space configuration
MAX_DICE_VALUE = 6  # Maximum face value on a die (1-6)
MAX_BID_AMOUNT = NUM_DICE_PER_PLAYER * NUM_PLAYERS  # Maximum amount in a bid (all dice same value)

# Derived constants
MAX_BID_POSITION = MAX_DICE_VALUE * MAX_BID_AMOUNT  # Total number of possible bids


```

Updated `MAX_BID_AMOUNT` to be derived from the number of dice:

**Before:** `MAX_BID_AMOUNT = 20` (arbitrary, too high)
**After:** `MAX_BID_AMOUNT = NUM_DICE_PER_PLAYER * NUM_PLAYERS = 4` (actual maximum)

This makes sense because:

- With 2 dice per player × 2 players = 4 total dice
- Maximum possible bid amount = 4 (if all 4 dice show the same value)
- You can't bid more than the total number of dice

**Impact:**

- Before: 6 values × 20 amounts = 120 possible bids
- After: 6 values × 4 amounts = 24 possible bids

This reduces the state space by about 5x, which should speed up policy generation. The bid space is now bounded by the actual game constraints rather than an arbitrary limit.
