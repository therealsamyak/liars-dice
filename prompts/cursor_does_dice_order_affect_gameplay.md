# Does dice order affect gameplay?

_Exported on 11/20/2025 at 11:44:26 PST from Cursor (2.0.77)_

---

**User**

does the dice order within the hand matter in this or not

---

**Cursor**

Checking how dice hands are represented and used to determine if order matters.

Checking the DP solver to confirm how it handles dice order:

**Dice order does not matter** in this implementation.

1. **State keys normalize by sorting**: Both `state_key` functions in `policy.py` and `dp_solver.py` sort the hand before creating the key:

```12:22:src/policy.py
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
```

2. **Counting only uses values**: The `count_dice` function in `rules.py` counts occurrences, not positions:

```6:8:src/rules.py
def count_dice(H: list[int], v: int) -> int:
    """COUNT(H, v): returns the number of dice in hand H showing value v."""
    return sum(1 for die in H if die == v)
```

3. **Game logic is order-agnostic**: Only the count of each value matters for bids and win conditions, not positions.

So `[1, 2, 3]` and `[3, 1, 2]` are treated as the same hand.

```python
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
```

```python
def count_dice(H: list[int], v: int) -> int:
    """COUNT(H, v): returns the number of dice in hand H showing value v."""
    return sum(1 for die in H if die == v)
```

---

**User**

make number of simulations a parameter in parameters.py, with half (rounded down) on truth telling policy and the remaining on against itself optimal policy

within those categories, make half (rounded down) with player 1 starting, and remaining player 2 starting, instead of random on who starts

---

**Cursor**

Reviewing the simulation code to understand how it's structured:

Updating the code to add the parameter and implement the simulation split:

Implementing the changes:

```python
"""System parameters for Liar's Dice simulation.

All system-wide constants are defined here and imported by other modules.
"""

# Dice configuration
NUM_DICE_PER_PLAYER = 4
NUM_PLAYERS = 2  # Number of players in the game

# Bid space configuration
MAX_DICE_VALUE = 6  # Maximum face value on a die (1-6)

# Policy configuration
LIAR_THRESHOLD = 0.01  # Probability threshold below which we call LIAR

# Simulation configuration
NUM_SIMULATIONS = 100  # Total number of simulations to run

# Derived constants
MAX_BID_AMOUNT = (
    NUM_DICE_PER_PLAYER * NUM_PLAYERS
)  # Maximum amount in a bid (all dice same value)
MAX_BID_POSITION = MAX_DICE_VALUE * MAX_BID_AMOUNT  # Total number of possible bids

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

    def simulate_game(self, verbose: bool = True, starting_player: int | None = None) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state).

        Args:
            verbose: Whether to print game details
            starting_player: 1 for Player 1, 2 for Player 2. If None, randomly chooses.
        """
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Choose starting player
        if starting_player is None:
            starting_player = random.randint(1, 2)
        else:
            starting_player = starting_player

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
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER, NUM_SIMULATIONS
from src.simulation import GameSimulation


def main():
    """Run simulations with optimal policy."""
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

    num_simulations = NUM_SIMULATIONS
    results_truth_telling = {"player1_wins": 0, "player2_wins": 0}
    results_optimal = {"player1_wins": 0, "player2_wins": 0}

    # Calculate simulation splits
    # Half (rounded down) on truth telling, remaining on optimal
    num_truth_telling = num_simulations // 2
    num_optimal = num_simulations - num_truth_telling

    # Within truth telling: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_tt_p1_start = num_truth_telling // 2
    num_tt_p2_start = num_truth_telling - num_tt_p1_start

    # Within optimal: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_opt_p1_start = num_optimal // 2
    num_opt_p2_start = num_optimal - num_opt_p1_start

    print(f"Running {num_simulations} simulations...")
    print(f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)")
    print(f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)")
    print("=" * 60)

    simulation_idx = 0

    # Run truth-telling simulations
    for i in range(num_truth_telling):
        starting_player = 1 if i < num_tt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="truth_telling",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if winner == 1:
            results_truth_telling["player1_wins"] += 1
        else:
            results_truth_telling["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    # Run optimal policy simulations
    for i in range(num_optimal):
        starting_player = 1 if i < num_opt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="optimal",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if winner == 1:
            results_optimal["player1_wins"] += 1
        else:
            results_optimal["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_truth_telling} with Player 2 using truth-telling policy")
    print(f"  - {num_optimal} with Player 2 using optimal policy")

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

Removing the redundant line in `simulation.py`:

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

    def simulate_game(self, verbose: bool = True, starting_player: int | None = None) -> tuple[int, State]:
        """Simulate a complete game. Returns (winner, final_state).

        Args:
            verbose: Whether to print game details
            starting_player: 1 for Player 1, 2 for Player 2. If None, randomly chooses.
        """
        # Initialize game state
        hand1 = self.roll_dice(self.num_dice_per_player)
        hand2 = self.roll_dice(self.num_dice_per_player)

        state = State(
            hand1=hand1, hand2=hand2, bid_history=[], game_status=GameStatus.ACTIVE
        )

        # Choose starting player
        if starting_player is None:
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

## Changes Made:

1. **Added `NUM_SIMULATIONS` parameter** to `parameters.py` (default: 100)

2. **Modified `simulation.py`** to accept a `starting_player` parameter (1 or 2) instead of randomly choosing

3. **Updated `main.py`** to:
   - Use `NUM_SIMULATIONS` from parameters
   - Split simulations: half (rounded down) vs truth-telling policy, remaining vs optimal policy
   - Within each category: half (rounded down) with Player 1 starting, remaining with Player 2 starting

Example with `NUM_SIMULATIONS = 100`:

- 50 vs truth-telling: 25 P1 starts, 25 P2 starts
- 50 vs optimal: 25 P1 starts, 25 P2 starts

The simulation output now shows which policy and starting player each simulation uses.

---

**User**

in the results section, also split it into those 2 sub sections per category

---

**Cursor**

Updating the results section to show subsections for each starting player within each policy category:

Updating the code to track results separately for each starting player scenario:

```python
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER, NUM_SIMULATIONS
from src.simulation import GameSimulation


def main():
    """Run simulations with optimal policy."""
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

    num_simulations = NUM_SIMULATIONS
    results_truth_telling_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_truth_telling_p2_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p2_start = {"player1_wins": 0, "player2_wins": 0}

    # Calculate simulation splits
    # Half (rounded down) on truth telling, remaining on optimal
    num_truth_telling = num_simulations // 2
    num_optimal = num_simulations - num_truth_telling

    # Within truth telling: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_tt_p1_start = num_truth_telling // 2
    num_tt_p2_start = num_truth_telling - num_tt_p1_start

    # Within optimal: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_opt_p1_start = num_optimal // 2
    num_opt_p2_start = num_optimal - num_opt_p1_start

    print(f"Running {num_simulations} simulations...")
    print(f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)")
    print(f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)")
    print("=" * 60)

    simulation_idx = 0

    # Run truth-telling simulations
    for i in range(num_truth_telling):
        starting_player = 1 if i < num_tt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="truth_telling",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if winner == 1:
            results_truth_telling["player1_wins"] += 1
        else:
            results_truth_telling["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    # Run optimal policy simulations
    for i in range(num_optimal):
        starting_player = 1 if i < num_opt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="optimal",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if winner == 1:
            results_optimal["player1_wins"] += 1
        else:
            results_optimal["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_truth_telling} with Player 2 using truth-telling policy")
    print(f"  - {num_optimal} with Player 2 using optimal policy")

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
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER, NUM_SIMULATIONS
from src.simulation import GameSimulation


def main():
    """Run simulations with optimal policy."""
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

    num_simulations = NUM_SIMULATIONS
    results_truth_telling_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_truth_telling_p2_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p2_start = {"player1_wins": 0, "player2_wins": 0}

    # Calculate simulation splits
    # Half (rounded down) on truth telling, remaining on optimal
    num_truth_telling = num_simulations // 2
    num_optimal = num_simulations - num_truth_telling

    # Within truth telling: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_tt_p1_start = num_truth_telling // 2
    num_tt_p2_start = num_truth_telling - num_tt_p1_start

    # Within optimal: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_opt_p1_start = num_optimal // 2
    num_opt_p2_start = num_optimal - num_opt_p1_start

    print(f"Running {num_simulations} simulations...")
    print(f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)")
    print(f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)")
    print("=" * 60)

    simulation_idx = 0

    # Run truth-telling simulations
    for i in range(num_truth_telling):
        starting_player = 1 if i < num_tt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="truth_telling",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if starting_player == 1:
            if winner == 1:
                results_truth_telling_p1_start["player1_wins"] += 1
            else:
                results_truth_telling_p1_start["player2_wins"] += 1
        else:
            if winner == 1:
                results_truth_telling_p2_start["player1_wins"] += 1
            else:
                results_truth_telling_p2_start["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    # Run optimal policy simulations
    for i in range(num_optimal):
        starting_player = 1 if i < num_opt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="optimal",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if winner == 1:
            results_optimal["player1_wins"] += 1
        else:
            results_optimal["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_truth_telling} with Player 2 using truth-telling policy")
    print(f"  - {num_optimal} with Player 2 using optimal policy")

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
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER, NUM_SIMULATIONS
from src.simulation import GameSimulation


def main():
    """Run simulations with optimal policy."""
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

    num_simulations = NUM_SIMULATIONS
    results_truth_telling_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_truth_telling_p2_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p2_start = {"player1_wins": 0, "player2_wins": 0}

    # Calculate simulation splits
    # Half (rounded down) on truth telling, remaining on optimal
    num_truth_telling = num_simulations // 2
    num_optimal = num_simulations - num_truth_telling

    # Within truth telling: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_tt_p1_start = num_truth_telling // 2
    num_tt_p2_start = num_truth_telling - num_tt_p1_start

    # Within optimal: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_opt_p1_start = num_optimal // 2
    num_opt_p2_start = num_optimal - num_opt_p1_start

    print(f"Running {num_simulations} simulations...")
    print(f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)")
    print(f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)")
    print("=" * 60)

    simulation_idx = 0

    # Run truth-telling simulations
    for i in range(num_truth_telling):
        starting_player = 1 if i < num_tt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="truth_telling",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if starting_player == 1:
            if winner == 1:
                results_truth_telling_p1_start["player1_wins"] += 1
            else:
                results_truth_telling_p1_start["player2_wins"] += 1
        else:
            if winner == 1:
                results_truth_telling_p2_start["player1_wins"] += 1
            else:
                results_truth_telling_p2_start["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    # Run optimal policy simulations
    for i in range(num_optimal):
        starting_player = 1 if i < num_opt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="optimal",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if starting_player == 1:
            if winner == 1:
                results_optimal_p1_start["player1_wins"] += 1
            else:
                results_optimal_p1_start["player2_wins"] += 1
        else:
            if winner == 1:
                results_optimal_p2_start["player1_wins"] += 1
            else:
                results_optimal_p2_start["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_truth_telling} with Player 2 using truth-telling policy")
    print(f"  - {num_optimal} with Player 2 using optimal policy")

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
"""Main entry point for Liar's Dice simulation."""

import os

from parameters import NUM_DICE_PER_PLAYER, NUM_SIMULATIONS
from src.simulation import GameSimulation


def main():
    """Run simulations with optimal policy."""
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

    num_simulations = NUM_SIMULATIONS
    results_truth_telling_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_truth_telling_p2_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p1_start = {"player1_wins": 0, "player2_wins": 0}
    results_optimal_p2_start = {"player1_wins": 0, "player2_wins": 0}

    # Calculate simulation splits
    # Half (rounded down) on truth telling, remaining on optimal
    num_truth_telling = num_simulations // 2
    num_optimal = num_simulations - num_truth_telling

    # Within truth telling: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_tt_p1_start = num_truth_telling // 2
    num_tt_p2_start = num_truth_telling - num_tt_p1_start

    # Within optimal: half (rounded down) with player 1 starting, remaining with player 2 starting
    num_opt_p1_start = num_optimal // 2
    num_opt_p2_start = num_optimal - num_opt_p1_start

    print(f"Running {num_simulations} simulations...")
    print(f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)")
    print(f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)")
    print("=" * 60)

    simulation_idx = 0

    # Run truth-telling simulations
    for i in range(num_truth_telling):
        starting_player = 1 if i < num_tt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="truth_telling",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if starting_player == 1:
            if winner == 1:
                results_truth_telling_p1_start["player1_wins"] += 1
            else:
                results_truth_telling_p1_start["player2_wins"] += 1
        else:
            if winner == 1:
                results_truth_telling_p2_start["player1_wins"] += 1
            else:
                results_truth_telling_p2_start["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    # Run optimal policy simulations
    for i in range(num_optimal):
        starting_player = 1 if i < num_opt_p1_start else 2
        simulation_idx += 1

        simulation = GameSimulation(
            num_dice_per_player=num_dice,
            player2_policy_type="optimal",
        )

        print(f"\n{'=' * 60}")
        print(f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)")
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(verbose=True, starting_player=starting_player)
        print(f"\n{'=' * 60}\n")

        if starting_player == 1:
            if winner == 1:
                results_optimal_p1_start["player1_wins"] += 1
            else:
                results_optimal_p1_start["player2_wins"] += 1
        else:
            if winner == 1:
                results_optimal_p2_start["player1_wins"] += 1
            else:
                results_optimal_p2_start["player2_wins"] += 1

        if simulation_idx % 10 == 0:
            print(f"Completed {simulation_idx}/{num_simulations} simulations...")

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTotal simulations: {num_simulations}")
    print(f"  - {num_truth_telling} with Player 2 using truth-telling policy")
    print(f"  - {num_optimal} with Player 2 using optimal policy")

    print("\n--- Results vs Truth-Telling Policy ---")
    total_tt = num_truth_telling
    total_tt_p1_start = num_tt_p1_start
    total_tt_p2_start = num_tt_p2_start

    print(f"\n  Player 1 starts ({total_tt_p1_start} simulations):")
    if total_tt_p1_start > 0:
        p1_wins_tt_p1 = results_truth_telling_p1_start["player1_wins"]
        p2_wins_tt_p1 = results_truth_telling_p1_start["player2_wins"]
        p1_win_rate_tt_p1 = p1_wins_tt_p1 / total_tt_p1_start * 100
        print(f"    Player 1 wins: {p1_wins_tt_p1}/{total_tt_p1_start} ({p1_win_rate_tt_p1:.1f}%)")
        print(f"    Player 2 wins: {p2_wins_tt_p1}/{total_tt_p1_start} ({100 - p1_win_rate_tt_p1:.1f}%)")

    print(f"\n  Player 2 starts ({total_tt_p2_start} simulations):")
    if total_tt_p2_start > 0:
        p1_wins_tt_p2 = results_truth_telling_p2_start["player1_wins"]
        p2_wins_tt_p2 = results_truth_telling_p2_start["player2_wins"]
        p1_win_rate_tt_p2 = p1_wins_tt_p2 / total_tt_p2_start * 100
        print(f"    Player 1 wins: {p1_wins_tt_p2}/{total_tt_p2_start} ({p1_win_rate_tt_p2:.1f}%)")
        print(f"    Player 2 wins: {p2_wins_tt_p2}/{total_tt_p2_start} ({100 - p1_win_rate_tt_p2:.1f}%)")

    print(f"\n  Total ({total_tt} simulations):")
    if total_tt > 0:
        total_p1_wins_tt = results_truth_telling_p1_start["player1_wins"] + results_truth_telling_p2_start["player1_wins"]
        total_p2_wins_tt = results_truth_telling_p1_start["player2_wins"] + results_truth_telling_p2_start["player2_wins"]
        total_p1_win_rate_tt = total_p1_wins_tt / total_tt * 100
        print(f"    Player 1 wins: {total_p1_wins_tt}/{total_tt} ({total_p1_win_rate_tt:.1f}%)")
        print(f"    Player 2 wins: {total_p2_wins_tt}/{total_tt} ({100 - total_p1_win_rate_tt:.1f}%)")

    print("\n--- Results vs Optimal Policy ---")
    total_opt = num_optimal
    total_opt_p1_start = num_opt_p1_start
    total_opt_p2_start = num_opt_p2_start

    print(f"\n  Player 1 starts ({total_opt_p1_start} simulations):")
    if total_opt_p1_start > 0:
        p1_wins_opt_p1 = results_optimal_p1_start["player1_wins"]
        p2_wins_opt_p1 = results_optimal_p1_start["player2_wins"]
        p1_win_rate_opt_p1 = p1_wins_opt_p1 / total_opt_p1_start * 100
        print(f"    Player 1 wins: {p1_wins_opt_p1}/{total_opt_p1_start} ({p1_win_rate_opt_p1:.1f}%)")
        print(f"    Player 2 wins: {p2_wins_opt_p1}/{total_opt_p1_start} ({100 - p1_win_rate_opt_p1:.1f}%)")

    print(f"\n  Player 2 starts ({total_opt_p2_start} simulations):")
    if total_opt_p2_start > 0:
        p1_wins_opt_p2 = results_optimal_p2_start["player1_wins"]
        p2_wins_opt_p2 = results_optimal_p2_start["player2_wins"]
        p1_win_rate_opt_p2 = p1_wins_opt_p2 / total_opt_p2_start * 100
        print(f"    Player 1 wins: {p1_wins_opt_p2}/{total_opt_p2_start} ({p1_win_rate_opt_p2:.1f}%)")
        print(f"    Player 2 wins: {p2_wins_opt_p2}/{total_opt_p2_start} ({100 - p1_win_rate_opt_p2:.1f}%)")

    print(f"\n  Total ({total_opt} simulations):")
    if total_opt > 0:
        total_p1_wins_opt = results_optimal_p1_start["player1_wins"] + results_optimal_p2_start["player1_wins"]
        total_p2_wins_opt = results_optimal_p1_start["player2_wins"] + results_optimal_p2_start["player2_wins"]
        total_p1_win_rate_opt = total_p1_wins_opt / total_opt * 100
        print(f"    Player 1 wins: {total_p1_wins_opt}/{total_opt} ({total_p1_win_rate_opt:.1f}%)")
        print(f"    Player 2 wins: {total_p2_wins_opt}/{total_opt} ({100 - total_p1_win_rate_opt:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()

```

Updated the results section to show subsections for each starting player within each policy category. The output now shows:

1. **Results vs Truth-Telling Policy**
   - Player 1 starts (with counts and percentages)
   - Player 2 starts (with counts and percentages)
   - Total (aggregated across both starting players)

2. **Results vs Optimal Policy**
   - Player 1 starts (with counts and percentages)
   - Player 2 starts (with counts and percentages)
   - Total (aggregated across both starting players)

This shows how the starting player affects win rates within each policy category.

---

**User**

does the generate polciy actually encode when to call liar in the json file based on the threshold?

---

**Cursor**

Checking how the policy generation works and whether it uses the threshold:

Reviewing the DP solver's decision logic:

Checking if the DP solver uses the threshold:

No. The policy generation does not use `LIAR_THRESHOLD` to decide when to call LIAR.

1. The DP solver (`dp_solver.py`) uses backward induction to choose actions. In `_compute_optimal_action`:

```307:345:src/dp_solver.py
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
```

It compares the expected value of calling LIAR vs making each legal bid, and picks the action with the highest expected value.

2. `LIAR_THRESHOLD` is only used as a fallback heuristic in `policy.py` when the policy isn't found in the JSON file:

```80:100:src/policy.py
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
```

3. The JSON policy file stores the optimal action for each state (either `None` for "call LIAR" or a bid dictionary), computed via backward induction, not a threshold.

In summary: the policy uses expected value maximization, not a threshold. The threshold is only used as a fallback when the policy file is missing.

```python
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
```

```python
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
```

---

**User**

wait then how is the policy solver determining the best move

---

**Cursor**

Reviewing the DP solver's decision logic:

The DP solver uses backward induction with expected value maximization. Process:

## 1. **Backward induction** (deepest to shallowest)

```190:202:src/dp_solver.py
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
```

Starts from terminal states (deepest bid histories) and works backward to initial states.

## 2. **Expected value of calling LIAR**

```347:371:src/dp_solver.py
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
```

Averages the reward over all possible opponent hands (uniform prior). Reward: +1.0 if Player 1 wins, -1.0 if Player 2 wins.

## 3. **Expected value of making a bid**

```373:409:src/dp_solver.py
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
```

For each possible opponent hand:

- If the opponent calls LIAR: compute immediate reward.
- If the opponent makes a bid: recursively use the already-computed value of the resulting state (via backward induction).

## 4. **Choose the best action**

```321:339:src/dp_solver.py
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
```

Selects the action (call LIAR or a specific bid) with the highest expected value.

## Summary

The solver:

1. Assumes the opponent uses `TruthTellingPolicy` (known behavior).
2. Uses backward induction to compute values from terminal states backward.
3. For each state, averages over all possible opponent hands (uniform prior).
4. Chooses the action that maximizes expected value (not a threshold).

This yields an optimal policy against a truth-telling opponent, assuming uniform prior over opponent hands.

```python
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
```

```python
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
```

```python
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
```

```python
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
```

---

**User**

shouldn't it use expected discounted reward such that later rewards are counted as worse

---

**Cursor**

Checking how the current implementation handles rewards:

The current implementation uses undiscounted rewards, which maximizes win probability regardless of game length. Whether to discount depends on the objective.

## Current Implementation (Undiscounted)

The code currently uses:

- **+1.0** if Player 1 wins
- **-1.0** if Player 2 wins

This maximizes expected win probability, treating a win in 2 turns the same as a win in 10 turns.

## Arguments for Discounting

1. Risk: Longer games increase uncertainty about opponent behavior.
2. Practical preference: Prefer shorter games.
3. Computational: Can help with convergence in some settings.

## Arguments Against Discounting

1. Zero-sum: The goal is maximizing win probability, not speed.
2. Finite horizon: The game always terminates (someone calls LIAR), so no infinite-horizon concern.
3. Optimality: Undiscounted maximizes win probability.

## If You Want to Add Discounting

You'd modify `_compute_bid_value` to multiply future values by a discount factor:

```python
# Instead of:
value = self._compute_optimal_action(new_obs)
total_value += value * self.hand_prob

# You'd do:
discount_factor = 0.95  # or whatever gamma you want
value = self._compute_optimal_action(new_obs)
total_value += discount_factor * value * self.hand_prob
```

And similarly propagate the discount through recursive calls.

**Question**: Do you want to maximize win probability (current) or prefer shorter games (discounted)? For competitive play, maximizing win probability is typically the goal, so undiscounted is appropriate. If you want to prefer shorter games, add discounting.

```python
# Instead of:
value = self._compute_optimal_action(new_obs)
total_value += value * self.hand_prob

# You'd do:
discount_factor = 0.95  # or whatever gamma you want
value = self._compute_optimal_action(new_obs)
total_value += discount_factor * value * self.hand_prob
```

---

**User**

set gamma within parameters.py, and use that

also reset the dice back to 2 again

---

**Cursor**

Adding a gamma parameter and resetting dice to 2. Updating the DP solver to use discounting:

Updating parameters and the DP solver:

```python
"""System parameters for Liar's Dice simulation.

All system-wide constants are defined here and imported by other modules.
"""

# Dice configuration
NUM_DICE_PER_PLAYER = 2
NUM_PLAYERS = 2  # Number of players in the game

# Bid space configuration
MAX_DICE_VALUE = 6  # Maximum face value on a die (1-6)

# Policy configuration
LIAR_THRESHOLD = 0.01  # Probability threshold below which we call LIAR
GAMMA = 0.95  # Discount factor for future rewards (0 < gamma <= 1)

# Simulation configuration
NUM_SIMULATIONS = 1000  # Total number of simulations to run

# Derived constants
MAX_BID_AMOUNT = (
    NUM_DICE_PER_PLAYER * NUM_PLAYERS
)  # Maximum amount in a bid (all dice same value)
MAX_BID_POSITION = MAX_DICE_VALUE * MAX_BID_AMOUNT  # Total number of possible bids

```

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA
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

```

Applying discounting to future rewards:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA
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
                # Apply discount factor to future rewards
                value = self._compute_optimal_action(new_obs)
                total_value += GAMMA * value * self.hand_prob

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

Discounting immediate rewards when the opponent calls LIAR, since they occur one step ahead:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA
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
                # Apply discount factor since this reward occurs one step in the future
                state = State(
                    hand1=obs.hand1.copy(),
                    hand2=hand2,
                    bid_history=new_bid_history.copy(),
                    game_status=GameStatus.GAME_OVER,
                )
                reward = compute_reward(state, caller=2, last_bid=bid)
                total_value += GAMMA * reward * self.hand_prob
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
                # Apply discount factor to future rewards
                value = self._compute_optimal_action(new_obs)
                total_value += GAMMA * value * self.hand_prob

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

## Changes Made:

1. **Added `GAMMA` parameter** to `parameters.py` (set to 0.95)
2. **Reset `NUM_DICE_PER_PLAYER`** back to 2
3. **Updated `dp_solver.py`** to:
   - Import `GAMMA` from parameters
   - Apply discounting to future rewards in `_compute_bid_value`:
     - When opponent calls LIAR: discount the immediate reward
     - When opponent makes a bid: discount the future state value

The discount factor is now applied consistently. Future rewards are discounted by `GAMMA` (0.95 by default), so the policy prefers shorter games when expected values are similar.

You can adjust `GAMMA` in `parameters.py`:

- `GAMMA = 1.0`: No discounting (maximize win probability regardless of game length)
- `GAMMA < 1.0`: Prefer shorter games (lower values = stronger preference for quick wins)

---

**User**

give an example for json output in the generate policy file when i run it, like how does it encode what move to make and when to call liar, shouldn't it use our threshold cause if the opponent hand is probability under that threshold it shouldc all liar

---

**Cursor**

Checking the JSON policy format and explaining how it encodes actions:

Explaining the JSON format and how the DP solver decides when to call LIAR:

## JSON Policy Format Example

The JSON file maps state keys to actions. Format:

**State Key Format:** `"hand;bid_history;status"`

- `hand`: sorted dice values (e.g., `"1,1"` = two dice showing 1)
- `bid_history`: pipe-separated bids (e.g., `"1,1|2,1|3,1"` = three bids)
- `status`: `"Active"` or `"GameOver"`

**Action Values:**

- `null` = **Call LIAR**
- `{"value": X, "amount": Y}` = **Make a bid** of Y dice showing value X

**Example from the file:**

```json
{
  "1,1;1,1|2,1|3,1|4,1|5,1|6,1|6,2|6,3|6,4|6,5;Active": null,
  "1,1;1,1|2,1|3,1|4,1|5,1|6,1;Active": {
    "value": 6,
    "amount": 2
  }
}
```

This means:

- State `"1,1;...|6,5;Active"`: **Call LIAR** (`null`)
- State `"1,1;1,1|2,1|3,1|4,1|5,1|6,1;Active"`: **Bid "2 sixes"** (`{"value": 6, "amount": 2}`)

## Why It Doesn't Use the Threshold

The DP solver does not use `LIAR_THRESHOLD` to decide when to call LIAR. It uses optimal value maximization:

```321:339:src/dp_solver.py
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
```

It:

1. Computes the expected value of calling LIAR (averages over all possible opponent hands)
2. Computes the expected value of each legal bid (considering future opponent responses)
3. Chooses the action with the highest expected value

This is optimal against a truth-telling opponent, not a probability threshold.

The `LIAR_THRESHOLD` is only used in the heuristic fallback when a state isn't found in the policy file:

```95:100:src/policy.py
        # Calculate probability that the last bid is valid
        prob_valid = p_valid(last_bid, obs.hand1, self.num_dice_per_player)

        # If probability is very low, call LIAR
        if prob_valid < LIAR_THRESHOLD:
            return None
```

The DP solver computes optimal actions via backward induction, not threshold-based heuristics.

```json
{
  "1,1;1,1|2,1|3,1|4,1|5,1|6,1|6,2|6,3|6,4|6,5;Active": null,
  "1,1;1,1|2,1|3,1|4,1|5,1|6,1;Active": {
    "value": 6,
    "amount": 2
  }
}
```

```python
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
```

```python
        # Calculate probability that the last bid is valid
        prob_valid = p_valid(last_bid, obs.hand1, self.num_dice_per_player)

        # If probability is very low, call LIAR
        if prob_valid < LIAR_THRESHOLD:
            return None
```

---

**User**

but the code you showed just says "CALL LIAR if ..." explain that

---

**Cursor**

Explaining how the DP solver decides when to call LIAR:

Tracing the logic that determines when calling LIAR is optimal:

## How the DP Solver Decides When to Call LIAR

The decision is based on expected value comparison, not a threshold. Here's the logic:

### Step 1: Compute Expected Value of Calling LIAR

```347:371:src/dp_solver.py
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
```

This function:

1. Iterates over all possible opponent hands (uniform prior)
2. For each hand, checks if the last bid is valid:
   - If COUNT(hand1 ∪ hand2, bid_value) >= bid_amount → bid is valid → Player 2 wins → reward = -1.0
   - If COUNT(hand1 ∪ hand2, bid_value) < bid_amount → bid is invalid → Player 1 wins → reward = +1.0
3. Averages these rewards: `expected_value = Σ (reward × probability)`

Example: If you have `[1,1]` and the last bid is "3 fours":

- Most opponent hands make the bid invalid → you win → +1.0
- Some opponent hands make it valid → you lose → -1.0
- The average might be, say, +0.3 (favoring calling LIAR)

### Step 2: Compute Expected Value of Making Each Bid

For each legal bid, it computes the expected value considering:

- All possible opponent hands
- What the opponent would do (using TruthTellingPolicy)
- Future game states (with discount factor)

### Step 3: Choose the Best Action

```321:339:src/dp_solver.py
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
```

The condition is: Call LIAR if `liar_value > max(bid_value for all legal bids)`

### In Practice

The solver calls LIAR when:

- The expected value of calling LIAR is higher than making any bid
- This depends on:
  - Your hand (what you know)
  - The bid history (what's been claimed)
  - The probability distribution over opponent hands
  - The opponent's policy (truth-telling)
  - The discount factor (preference for shorter games)

It's not "if probability < threshold, call LIAR" — it's "if calling LIAR has higher expected value than continuing, call LIAR."

The threshold (`LIAR_THRESHOLD`) is only used in the heuristic fallback when a state isn't in the precomputed policy.

```python
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
```

```python
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
```

---

**User**

how does it compute the expected value of calling liar

---

**Cursor**

Tracing how `_compute_liar_value` computes the expected value with a concrete example:

Tracing the computation with a concrete example:

## Example: Computing Expected Value of Calling LIAR

**Setup:**

- Player 1 has hand: `[1, 1]` (two dice showing 1)
- Last bid: "3 fours" (value=4, amount=3)
- 2 dice per player = 36 possible opponent hands

### Step-by-Step Computation:

```347:371:src/dp_solver.py
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
```

**For each opponent hand:**

1. **Opponent has `[1, 1]`:**
   - Player 1 has 0 fours, opponent has 0 fours
   - Total fours = 0 < 3 → Bid is invalid → Player 1 wins → reward = +1.0

2. **Opponent has `[4, 1]`:**
   - Player 1 has 0 fours, opponent has 1 four
   - Total fours = 1 < 3 → Bid is invalid → Player 1 wins → reward = +1.0

3. **Opponent has `[4, 4]`:**
   - Player 1 has 0 fours, opponent has 2 fours
   - Total fours = 2 < 3 → Bid is invalid → Player 1 wins → reward = +1.0

4. **Opponent has `[4, 4, ...]` (impossible with 2 dice, but if they had 3+ fours):**
   - Total fours ≥ 3 → Bid is valid → Player 2 wins → reward = -1.0

**Averaging:**

- With 2 dice, the opponent can have at most 2 fours
- So total fours = 0 (from Player 1) + 0-2 (from opponent) = 0-2
- Since 0-2 < 3, the bid is always invalid
- All 36 hands give reward = +1.0
- Expected value = (36 × +1.0) / 36 = +1.0

**More realistic example:**

If Player 1 has `[4, 4]` (two fours) and the last bid is "3 fours":

- Opponent has `[1, 1]`: Total = 2 + 0 = 2 < 3 → +1.0
- Opponent has `[4, 1]`: Total = 2 + 1 = 3 ≥ 3 → -1.0
- Opponent has `[4, 4]`: Total = 2 + 2 = 4 ≥ 3 → -1.0
- ... (continue for all 36 hands)

Expected value = (wins × +1.0 + losses × -1.0) / 36

### The Key Formula:

```
Expected Value = Σ (over all opponent hands) [P(opponent_hand) × Reward(opponent_hand)]
```

Where:

- `P(opponent_hand) = 1/36` (uniform prior, since `hand_prob = 1.0 / num_hands`)
- `Reward(opponent_hand) = +1.0` if Player 1 wins, `-1.0` if Player 2 wins
- Win condition: `COUNT(hand1 ∪ hand2, bid_value) < bid_amount` → Player 1 wins

The function sums `reward × hand_prob` for all possible opponent hands, which gives the expected value of calling LIAR.

```python
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
```

```plaintext
Expected Value = Σ (over all opponent hands) [P(opponent_hand) × Reward(opponent_hand)]
```

---

**User**

@zsh (986-1024) is this expected? or did something go wrong

---

**Cursor**

Checking how Player 2 uses the optimal policy and whether there's a first-mover advantage:

There's a mismatch: the policy was trained assuming the opponent uses a truth-telling policy, but in the "optimal vs optimal" case, both players use the same policy. Checking how the policy was generated:

There’s a mismatch: the policy was trained assuming the opponent uses a truth-telling policy, but in the “optimal vs optimal” case both players use the same policy.

Looking at the results:

1. First-mover advantage: whoever starts wins ~98–100% of the time.
2. Against truth-telling: Player 1 wins 98% when starting, 16% when Player 2 starts.
3. Optimal vs optimal: Player 1 wins 100% when starting, 2% when Player 2 starts.

This is expected because:

1. The policy is optimal against a truth-telling opponent, not against an optimal opponent.
2. There’s a strong first-mover advantage in Liar’s Dice with optimal play.

The asymmetry comes from:

- The policy assumes the opponent is truth-telling.
- When both players use it, the assumptions don’t hold.
- The first mover can exploit this.

To test optimal vs optimal correctly, we’d need a policy computed assuming the opponent also uses optimal play (e.g., via self-play or a Nash equilibrium solver). The current policy is optimal against truth-telling, which explains the results.

The strong first-mover advantage (98–100%) is likely real in this game with optimal play, not a bug.

---

**User**

what do you mean that our policy is optimal only against truth telling opponent, our @MATH.md doesn't mention a specific opponent our strategy should be very general

---

**Cursor**

Reviewing MATH.md and the DP solver to clarify:

There’s a mismatch: MATH.md doesn’t specify an opponent policy, but the implementation assumes truth-telling.

Looking at the code:

```20:26:generate_policy.py
    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(
        num_dice, opponent_policy, max_value=MAX_DICE_VALUE, max_amount=MAX_BID_AMOUNT
    )
```

And in the DP solver:

```382:383:src/dp_solver.py
            # Get opponent's action given their hand and new bid history
            opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)
```

The solver uses `opponent_policy` to predict opponent actions. This is an implementation assumption not in MATH.md.

MATH.md says:

- "Modeled from Player 1 perspective. Opponent actions handled by system internally."
- Uses uniform prior over opponent hands: `$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$`
- Doesn’t specify an opponent policy

To compute a general optimal policy, we should solve for a Nash equilibrium (both players optimal), not assume truth-telling. The current approach computes optimal play against a truth-telling opponent, which explains the simulation results.

Options:

1. Keep the current approach (optimal vs truth-telling) and document it.
2. Implement Nash equilibrium solving (harder, more general).
3. Make the opponent policy configurable so different assumptions can be tested.

The strong first-mover advantage in the results is likely real, but the policy is optimal only against truth-telling, not against optimal play.

```python
    # Create opponent policy
    opponent_policy = TruthTellingPolicy(num_dice)

    # Create DP solver
    solver = DPSolver(
        num_dice, opponent_policy, max_value=MAX_DICE_VALUE, max_amount=MAX_BID_AMOUNT
    )
```

```python
            # Get opponent's action given their hand and new bid history
            opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)
```

---

**User**

the opponent policy is unknown remember, thats the whole point their hand could be any dice that we don't know, hence why in parameters we have a liar threshold so based on our hand and the total number of dice as referenced in math.md implement that

---

**Cursor**

Updating the DP solver to use the probability-based approach from MATH.md instead of assuming a specific opponent policy. Checking how `P_valid` is computed:

Updating the DP solver to use probability-based reasoning from MATH.md instead of assuming a specific opponent policy. Implementing a threshold-based opponent model:

Updating the DP solver to use probability-based reasoning from MATH.md instead of assuming a specific opponent policy:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA, LIAR_THRESHOLD
from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.probability import p_valid
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
                # Apply discount factor since this reward occurs one step in the future
                state = State(
                    hand1=obs.hand1.copy(),
                    hand2=hand2,
                    bid_history=new_bid_history.copy(),
                    game_status=GameStatus.GAME_OVER,
                )
                reward = compute_reward(state, caller=2, last_bid=bid)
                total_value += GAMMA * reward * self.hand_prob
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
                # Apply discount factor to future rewards
                value = self._compute_optimal_action(new_obs)
                total_value += GAMMA * value * self.hand_prob

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
    def __init__(
        self,
        num_dice_per_player: int,
        opponent_policy: TruthTellingPolicy | None = None,
        max_value: int = MAX_DICE_VALUE,
        max_amount: int = MAX_BID_AMOUNT,
    ):
        self.num_dice_per_player = num_dice_per_player
        # Opponent policy only used as fallback for bid selection when P_valid >= threshold
        # Primary decision (LIAR vs bid) uses probability-based threshold from MATH.md
        self.opponent_policy = opponent_policy or TruthTellingPolicy(num_dice_per_player)
        self.max_value = max_value
        self.max_amount = max_amount
        self.value_function: dict[str, float] = {}
        self.policy: dict[str, dict | None] = {}  # None means call LIAR
        self.all_hands = generate_all_hands(num_dice_per_player)
        self.num_hands = len(self.all_hands)
        self.hand_prob = 1.0 / self.num_hands  # Uniform prior
        self.max_depth = get_bid_position(
            max_value, max_amount
        )  # Total number of possible bids
```

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA, LIAR_THRESHOLD
from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.probability import p_valid
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
        """Compute expected value of making a bid, marginalizing over opponent's hand.

        Uses probability-based reasoning from MATH.md: opponent calls LIAR if
        P_valid < LIAR_THRESHOLD, otherwise makes a bid (using fallback policy).
        """
        new_bid_history = obs.bid_history + [bid]
        total_value = 0.0

        # Marginalize over all possible opponent hands (uniform prior)
        for hand2_tuple in self.all_hands:
            hand2 = list(hand2_tuple)

            # Compute P_valid from opponent's perspective (they know their hand)
            # P_valid = P(COUNT(H1 ∪ H2, bid.value) >= bid.amount | H2)
            prob_valid = p_valid(bid, hand2, self.num_dice_per_player)

            if prob_valid < LIAR_THRESHOLD:
                # Opponent calls LIAR based on probability threshold (MATH.md approach)
                # Apply discount factor since this reward occurs one step in the future
                state = State(
                    hand1=obs.hand1.copy(),
                    hand2=hand2,
                    bid_history=new_bid_history.copy(),
                    game_status=GameStatus.GAME_OVER,
                )
                reward = compute_reward(state, caller=2, last_bid=bid)
                total_value += GAMMA * reward * self.hand_prob
            else:
                # Opponent makes a bid (P_valid >= threshold)
                # Use fallback policy to determine which bid they make
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

                if opponent_action is None:
                    # Fallback policy says call LIAR, but P_valid >= threshold
                    # This shouldn't happen often, but handle it as terminal state
                    state = State(
                        hand1=obs.hand1.copy(),
                        hand2=hand2,
                        bid_history=new_bid_history.copy(),
                        game_status=GameStatus.GAME_OVER,
                    )
                    reward = compute_reward(state, caller=2, last_bid=bid)
                    total_value += GAMMA * reward * self.hand_prob
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
                    # Apply discount factor to future rewards
                    value = self._compute_optimal_action(new_obs)
                    total_value += GAMMA * value * self.hand_prob

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

Fixing the `__init__` method:

```python
"""Dynamic Programming solver for optimal Liar's Dice policy.

This solver enumerates all possible observable states and computes optimal actions
using backward induction, marginalizing over opponent's unknown hand.
"""

import json
from collections import defaultdict
from itertools import product

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, GAMMA, LIAR_THRESHOLD
from src.game_state import Bid, GameStatus, ObservableState, State
from src.opponent_policies import TruthTellingPolicy
from src.probability import p_valid
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
        opponent_policy: TruthTellingPolicy | None = None,
        max_value: int = MAX_DICE_VALUE,
        max_amount: int = MAX_BID_AMOUNT,
    ):
        self.num_dice_per_player = num_dice_per_player
        # Opponent policy only used as fallback for bid selection when P_valid >= threshold
        # Primary decision (LIAR vs bid) uses probability-based threshold from MATH.md
        self.opponent_policy = opponent_policy or TruthTellingPolicy(num_dice_per_player)
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
        """Compute expected value of making a bid, marginalizing over opponent's hand.

        Uses probability-based reasoning from MATH.md: opponent calls LIAR if
        P_valid < LIAR_THRESHOLD, otherwise makes a bid (using fallback policy).
        """
        new_bid_history = obs.bid_history + [bid]
        total_value = 0.0

        # Marginalize over all possible opponent hands (uniform prior)
        for hand2_tuple in self.all_hands:
            hand2 = list(hand2_tuple)

            # Compute P_valid from opponent's perspective (they know their hand)
            # P_valid = P(COUNT(H1 ∪ H2, bid.value) >= bid.amount | H2)
            prob_valid = p_valid(bid, hand2, self.num_dice_per_player)

            if prob_valid < LIAR_THRESHOLD:
                # Opponent calls LIAR based on probability threshold (MATH.md approach)
                # Apply discount factor since this reward occurs one step in the future
                state = State(
                    hand1=obs.hand1.copy(),
                    hand2=hand2,
                    bid_history=new_bid_history.copy(),
                    game_status=GameStatus.GAME_OVER,
                )
                reward = compute_reward(state, caller=2, last_bid=bid)
                total_value += GAMMA * reward * self.hand_prob
            else:
                # Opponent makes a bid (P_valid >= threshold)
                # Use fallback policy to determine which bid they make
                opponent_action = self.opponent_policy.get_action(hand2, new_bid_history)

                if opponent_action is None:
                    # Fallback policy says call LIAR, but P_valid >= threshold
                    # This shouldn't happen often, but handle it as terminal state
                    state = State(
                        hand1=obs.hand1.copy(),
                        hand2=hand2,
                        bid_history=new_bid_history.copy(),
                        game_status=GameStatus.GAME_OVER,
                    )
                    reward = compute_reward(state, caller=2, last_bid=bid)
                    total_value += GAMMA * reward * self.hand_prob
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
                    # Apply discount factor to future rewards
                    value = self._compute_optimal_action(new_obs)
                    total_value += GAMMA * value * self.hand_prob

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

Updating `generate_policy.py` to reflect that opponent policy is now optional and only used as a fallback:

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

    # Create DP solver
    # Opponent policy is optional - if None, uses probability-based threshold from MATH.md
    # TruthTellingPolicy is only used as fallback for bid selection when P_valid >= threshold
    solver = DPSolver(
        num_dice, opponent_policy=None, max_value=MAX_DICE_VALUE, max_amount=MAX_BID_AMOUNT
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

Removing the unused import:

```python
"""Generate optimal DP policy for Liar's Dice."""

import os

from parameters import MAX_BID_AMOUNT, MAX_DICE_VALUE, NUM_DICE_PER_PLAYER
from src.dp_solver import DPSolver


def main():
    """Generate and save optimal policy."""
    num_dice = NUM_DICE_PER_PLAYER

    print("=" * 60)
    print("Generating Optimal DP Policy for Liar's Dice")
    print("=" * 60)
    print(f"Number of dice per player: {num_dice}")
    print()

    # Create DP solver
    # Opponent policy is optional - if None, uses probability-based threshold from MATH.md
    # TruthTellingPolicy is only used as fallback for bid selection when P_valid >= threshold
    solver = DPSolver(
        num_dice, opponent_policy=None, max_value=MAX_DICE_VALUE, max_amount=MAX_BID_AMOUNT
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

Updated the DP solver to use probability-based reasoning from MATH.md instead of assuming a specific opponent policy.

## Changes Made:

1. **Updated `_compute_bid_value`** to use probability-based thresholding:
   - For each possible opponent hand, computes `P_valid` from their perspective using the formula from MATH.md
   - If `P_valid < LIAR_THRESHOLD`: opponent calls LIAR (probability-based decision)
   - If `P_valid >= LIAR_THRESHOLD`: opponent makes a bid (uses truth-telling policy as fallback to determine which bid)

2. **Made opponent policy optional** in `DPSolver.__init__`:
   - If `None`, creates a default `TruthTellingPolicy` as fallback
   - The fallback is only used to determine which bid to make when `P_valid >= threshold`
   - The primary decision (LIAR vs bid) uses probability-based reasoning

3. **Updated `generate_policy.py`** to reflect that opponent policy is optional

The policy now follows MATH.md:

- Uses uniform prior over opponent hands: `$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$`
- Uses probability-based thresholding: `$P_{valid}(D_{last} \mid H_1)$` with `LIAR_THRESHOLD`
- Doesn't assume a specific opponent policy; only uses it as a fallback for bid selection

This aligns with the math formulation where the opponent's policy is unknown and we reason probabilistically.
