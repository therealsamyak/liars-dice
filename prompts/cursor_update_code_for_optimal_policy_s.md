# Update code for optimal policy simulation

_Exported on 11/20/2025 at 11:01:12 PST from Cursor (2.0.77)_

---

**User**

using @MATH.md as a baseline, update the code to be accurate if its not already

i want it to solve the policy in generate_policy such that i get a .json file with the optimal 'dp matrix' type solution that tells the player 1 its optimal move at every timestep

then, the main.py should run 100 simulations each with the player 1 (using the policy) against player 2 (truth telling policy) and player 2 (using the same policy), with randomness on who goes first in each simulation

fix the code accordingly

store all policy's generated in a new folder policy/

to make this quick, the def

---

**Cursor**

Reviewing the codebase for accuracy against MATH.md, then updating `generate_policy.py` and `main.py` accordingly.

Creating a todo list and updating the code:

Updating the code:

```python
"""Generate optimal DP policy for Liar's Dice."""

import os

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
"""Optimal policy implementation for Player 1."""

import json
import os
from collections import Counter

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
            policy_file = os.path.join("policy", f"policy_{num_dice_per_player}dice.json")

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

        print(f"Loaded policy from {filepath} ({len(self.policy)} states)")

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
        if prob_valid < 0.1:
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

```

Updating `simulation.py` to support both policy types for Player 2, then updating `main.py`:

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

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

```python
"""Main entry point for Liar's Dice simulation."""

import os
import random

from src.simulation import GameSimulation


def main():
    """Run 100 simulations with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    num_dice = 2
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
    total_tt = results_truth_telling["player1_wins"] + results_truth_telling["player2_wins"]
    if total_tt > 0:
        p1_win_rate_tt = results_truth_telling["player1_wins"] / total_tt * 100
        print(f"Player 1 wins: {results_truth_telling['player1_wins']}/{total_tt} ({p1_win_rate_tt:.1f}%)")
        print(f"Player 2 wins: {results_truth_telling['player2_wins']}/{total_tt} ({100 - p1_win_rate_tt:.1f}%)")

    print("\n--- Results vs Optimal Policy ---")
    total_opt = results_optimal["player1_wins"] + results_optimal["player2_wins"]
    if total_opt > 0:
        p1_win_rate_opt = results_optimal["player1_wins"] / total_opt * 100
        print(f"Player 1 wins: {results_optimal['player1_wins']}/{total_opt} ({p1_win_rate_opt:.1f}%)")
        print(f"Player 2 wins: {results_optimal['player2_wins']}/{total_opt} ({100 - p1_win_rate_opt:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()

```

Reviewing the simulation logic to ensure it handles Player 2 starting correctly:

Adding a safety check for when Player 2 starts and there's no bid history:

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print("Player 2 cannot call LIAR with no bids, making minimal bid")
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

```python
"""Game simulation engine."""

import random

from src.game_state import GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                            print("Player 1 cannot call LIAR with no bids, making minimal bid")
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print("Player 2 cannot call LIAR with no bids, making minimal bid")
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

```python
"""Game simulation engine."""

import random

from src.game_state import Bid, GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                            print("Player 1 cannot call LIAR with no bids, making minimal bid")
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print("Player 2 cannot call LIAR with no bids, making minimal bid")
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

Fixing the logic flow:

```python
"""Game simulation engine."""

import random

from src.game_state import Bid, GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                            print("Player 1 cannot call LIAR with no bids, making minimal bid")
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print("Player 2 cannot call LIAR with no bids, making minimal bid")
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

```python
"""Game simulation engine."""

import random

from src.game_state import Bid, GameStatus, ObservableState, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(
        self,
        num_dice_per_player: int = 2,
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
                            print("Player 1 cannot call LIAR with no bids, making minimal bid")
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
                    action = self.player2_policy.get_action(state.hand2, state.bid_history)

                if action is None:
                    # Player 2 calls LIAR
                    if not state.bid_history:
                        # Should not happen - can't call LIAR with no bids
                        # Fallback: make a minimal bid
                        action = Bid(value=1, amount=1)
                        if verbose:
                            print("Player 2 cannot call LIAR with no bids, making minimal bid")
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

Verifying code accuracy against MATH.md and checking for issues:

Checking for linting errors:

Verifying the code matches MATH.md. Checking one more detail about the DP solver's reward computation:

## Summary of Changes

1. **`generate_policy.py`**: Updated to save policies to the `policy/` folder (creates the folder if it doesn't exist).

2. **`main.py`**: Updated to:
   - Run 100 simulations
   - Alternate Player 2 policy: 50 with truth-telling, 50 with optimal policy
   - Randomize starting player (already handled in simulation)
   - Print summary statistics

3. **`src/simulation.py`**: Updated to:
   - Accept `player2_policy_type` parameter ("truth_telling" or "optimal")
   - Support Player 2 using the optimal policy (creates observable state from their perspective)
   - Add safety checks for edge cases

4. **`src/policy.py`**: Updated default policy file path to look in the `policy/` folder.

## Code Accuracy Verification

Verified against MATH.md:

- Bidding rules match (increase value/amount, cannot decrease)
- Win conditions match (caller vs bidder logic)
- Probability calculations match (binomial distribution for P_valid)
- State representation matches (observable state O = (H1, D, Go))
- Reward function matches (+1 for P1 win, -1 for P2 win)

The code is ready to use. Run `python generate_policy.py` to generate the policy, then `python main.py` to run the 100 simulations.

---

**User**

change the bidding rules in math.md and in the code to match actual liar's dice

---

**Cursor**

Checking the current bidding rules and updating them to match standard Liar's Dice. Researching the standard rules first:

Updating the rules: when increasing face value, any quantity of the higher face value is allowed (even 1). Checking the current implementation:

Updating MATH.md and the code to match standard Liar's Dice: when increasing face value, any quantity of the higher face value is allowed (even 1).

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value
  - You can increase quantity while keeping same face value (e.g., "three 3s" → "four 3s")
  - You can increase face value with ANY quantity (e.g., "three 3s" → "one 4s" or "three 3s" → "five 4s")
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)
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

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value
  - You can increase quantity while keeping same face value (e.g., "three 3s" → "four 3s")
  - You can increase face value with ANY quantity (e.g., "three 3s" → "one 4s" or "three 3s" → "five 4s")
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)
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
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ (any amount allowed when increasing value)} \\
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

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value
  - You can increase quantity while keeping same face value (e.g., "three 3s" → "four 3s")
  - You can increase face value with ANY quantity (e.g., "three 3s" → "one 4s" or "three 3s" → "five 4s")
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)
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
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ (any amount allowed when increasing value)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

**Bid Priority Order:** When generating legal bids, priority is:

1. First: increase face value (any quantity allowed, ordered by value then amount, lowest first)
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
    - Each subsequent bid must increase either quantity OR face value
    - You can increase quantity while keeping same face value
    - You can increase face value with ANY quantity (even 1)
    - You CANNOT decrease face value
    - You CANNOT decrease quantity when keeping same face value

    Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current (any amount allowed when increasing value)
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
    - Each subsequent bid must increase either quantity OR face value
    - You can increase quantity while keeping same face value
    - You can increase face value with ANY quantity (even 1)
    - You CANNOT decrease face value
    - You CANNOT decrease quantity when keeping same face value

    Examples:
    - "three 3s" → "four 3s" ✓ valid (increase quantity, same value)
    - "three 3s" → "three 4s" ✓ valid (increase value, same quantity)
    - "three 3s" → "one 4s" ✓ valid (increase value, any quantity allowed)
    - "three 3s" → "five 4s" ✓ valid (increase both)
    - "three 3s" → "two 3s" ✗ invalid (decrease quantity, same value)
    - "four 4s" → "three 3s" ✗ invalid (decrease value)

    LEGAL(D) = {
        True  if D = ∅ (first bid, any valid bid allowed)
        True  if DiceValue_new > DiceValue_current (any amount allowed when increasing value)
        True  if DiceValue_new = DiceValue_current and Amount_new > Amount_current
        False otherwise
    }
    """
    if D_last is None:
        # D = ∅: first bid - any valid bid is allowed
        return True

    # DiceValue_new > DiceValue_current
    # When increasing value, any amount is allowed (standard Liar's Dice rule)
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

Updating `opponent_policies.py` to match the new rules:

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

        When increasing value, any amount is allowed (standard Liar's Dice rule).
        """
        # Try bids in priority order: increase value first, then increase amount
        # Priority: (value+1, 1), (value+1, 2), ..., (6, 1), (6, 2), ...
        #          then (value, amount+1), (value, amount+2), ...

        # First, try increasing value (any amount allowed when increasing value)
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

        When increasing value, any amount is allowed (standard Liar's Dice rule).
        """
        # Try bids in priority order: increase value first, then increase amount
        # Priority: (value+1, 1), (value+1, 2), ..., (6, 1), (6, 2), ...
        #          then (value, amount+1), (value, amount+2), ...

        # First, try increasing value (any amount allowed when increasing value)
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

        Priority order: increase value first (any amount allowed), then increase amount.
        When increasing value, any amount is allowed (standard Liar's Dice rule).
        """
        # Try increasing value with minimum amount (1)
        for value in range(last_bid.value + 1, 7):
            bid = Bid(value=value, amount=1)
            if is_legal_bid(bid, last_bid):
                return bid

        # Then try same value with increased amount
        return Bid(value=last_bid.value, amount=last_bid.amount + 1)

```

Checking for other references to the old bidding rules:

Verifying the changes are consistent:

## Summary of Changes

**Updated bidding rules to match standard Liar's Dice:**

1. **MATH.md**: Updated documentation to reflect that when increasing face value, any quantity is allowed (even 1).

2. **src/rules.py**: Updated `is_legal_bid()` to allow any amount when increasing face value (removed the `amount >= current amount` requirement).

3. **src/dp_solver.py**: Updated `generate_all_legal_bids()` to generate all amounts (starting from 1) when increasing face value.

4. **src/opponent_policies.py**: Updated truth-telling policy to try amounts starting from 1 when increasing face value.

**Key change:**

- **Old rule**: "three 3s" → "two 4s" was invalid (couldn't decrease quantity when increasing value)
- **New rule**: "three 3s" → "one 4s" is now valid (any quantity allowed when increasing value)

All files are consistent with the standard Liar's Dice rules. The code now allows any quantity when increasing face value, while still requiring an increase in quantity when keeping the same face value.

---

**User**

remove all weird unicode symbols like the checkmark or the x from math.md it should just be normal latex or normal english alphabet letters

---

**Cursor**

Removing Unicode symbols from MATH.md and replacing them with LaTeX or plain text.

Replacing Unicode symbols (✓, ✗, →) with plain text.

```markdown
## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value
  - You can increase quantity while keeping same face value (e.g., "three 3s" -> "four 3s")
  - You can increase face value with ANY quantity (e.g., "three 3s" -> "one 4s" or "three 3s" -> "five 4s")
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" -> "four 3s" [VALID] (increase quantity, same value)
    - "three 3s" -> "three 4s" [VALID] (increase value, same quantity)
    - "three 3s" -> "one 4s" [VALID] (increase value, any quantity allowed)
    - "three 3s" -> "five 4s" [VALID] (increase both)
    - "three 3s" -> "two 3s" [INVALID] (decrease quantity, same value)
    - "four 4s" -> "three 3s" [INVALID] (decrease value)
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
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ (any amount allowed when increasing value)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

**Bid Priority Order:** When generating legal bids, priority is:

1. First: increase face value (any quantity allowed, ordered by value then amount, lowest first)
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

Replaced Unicode symbols in MATH.md:

- ✓ → `[VALID]`
- ✗ → `[INVALID]`
- → → `->`

All examples now use plain ASCII text. The file should render correctly without Unicode dependencies.
