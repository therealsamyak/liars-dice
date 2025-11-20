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
