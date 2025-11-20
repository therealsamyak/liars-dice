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
    print(
        f"  - {num_truth_telling} vs truth-telling policy ({num_tt_p1_start} P1 start, {num_tt_p2_start} P2 start)"
    )
    print(
        f"  - {num_optimal} vs optimal policy ({num_opt_p1_start} P1 start, {num_opt_p2_start} P2 start)"
    )
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
        print(
            f"SIMULATION {simulation_idx}/{num_simulations} (Truth-Telling, P{starting_player} starts)"
        )
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(
            verbose=True, starting_player=starting_player
        )
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
        print(
            f"SIMULATION {simulation_idx}/{num_simulations} (Optimal, P{starting_player} starts)"
        )
        print(f"{'=' * 60}")
        winner, _ = simulation.simulate_game(
            verbose=True, starting_player=starting_player
        )
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
        print(
            f"    Player 1 wins: {p1_wins_tt_p1}/{total_tt_p1_start} ({p1_win_rate_tt_p1:.1f}%)"
        )
        print(
            f"    Player 2 wins: {p2_wins_tt_p1}/{total_tt_p1_start} ({100 - p1_win_rate_tt_p1:.1f}%)"
        )

    print(f"\n  Player 2 starts ({total_tt_p2_start} simulations):")
    if total_tt_p2_start > 0:
        p1_wins_tt_p2 = results_truth_telling_p2_start["player1_wins"]
        p2_wins_tt_p2 = results_truth_telling_p2_start["player2_wins"]
        p1_win_rate_tt_p2 = p1_wins_tt_p2 / total_tt_p2_start * 100
        print(
            f"    Player 1 wins: {p1_wins_tt_p2}/{total_tt_p2_start} ({p1_win_rate_tt_p2:.1f}%)"
        )
        print(
            f"    Player 2 wins: {p2_wins_tt_p2}/{total_tt_p2_start} ({100 - p1_win_rate_tt_p2:.1f}%)"
        )

    print(f"\n  Total ({total_tt} simulations):")
    if total_tt > 0:
        total_p1_wins_tt = (
            results_truth_telling_p1_start["player1_wins"]
            + results_truth_telling_p2_start["player1_wins"]
        )
        total_p2_wins_tt = (
            results_truth_telling_p1_start["player2_wins"]
            + results_truth_telling_p2_start["player2_wins"]
        )
        total_p1_win_rate_tt = total_p1_wins_tt / total_tt * 100
        print(
            f"    Player 1 wins: {total_p1_wins_tt}/{total_tt} ({total_p1_win_rate_tt:.1f}%)"
        )
        print(
            f"    Player 2 wins: {total_p2_wins_tt}/{total_tt} ({100 - total_p1_win_rate_tt:.1f}%)"
        )

    print("\n--- Results vs Optimal Policy ---")
    total_opt = num_optimal
    total_opt_p1_start = num_opt_p1_start
    total_opt_p2_start = num_opt_p2_start

    print(f"\n  Player 1 starts ({total_opt_p1_start} simulations):")
    if total_opt_p1_start > 0:
        p1_wins_opt_p1 = results_optimal_p1_start["player1_wins"]
        p2_wins_opt_p1 = results_optimal_p1_start["player2_wins"]
        p1_win_rate_opt_p1 = p1_wins_opt_p1 / total_opt_p1_start * 100
        print(
            f"    Player 1 wins: {p1_wins_opt_p1}/{total_opt_p1_start} ({p1_win_rate_opt_p1:.1f}%)"
        )
        print(
            f"    Player 2 wins: {p2_wins_opt_p1}/{total_opt_p1_start} ({100 - p1_win_rate_opt_p1:.1f}%)"
        )

    print(f"\n  Player 2 starts ({total_opt_p2_start} simulations):")
    if total_opt_p2_start > 0:
        p1_wins_opt_p2 = results_optimal_p2_start["player1_wins"]
        p2_wins_opt_p2 = results_optimal_p2_start["player2_wins"]
        p1_win_rate_opt_p2 = p1_wins_opt_p2 / total_opt_p2_start * 100
        print(
            f"    Player 1 wins: {p1_wins_opt_p2}/{total_opt_p2_start} ({p1_win_rate_opt_p2:.1f}%)"
        )
        print(
            f"    Player 2 wins: {p2_wins_opt_p2}/{total_opt_p2_start} ({100 - p1_win_rate_opt_p2:.1f}%)"
        )

    print(f"\n  Total ({total_opt} simulations):")
    if total_opt > 0:
        total_p1_wins_opt = (
            results_optimal_p1_start["player1_wins"]
            + results_optimal_p2_start["player1_wins"]
        )
        total_p2_wins_opt = (
            results_optimal_p1_start["player2_wins"]
            + results_optimal_p2_start["player2_wins"]
        )
        total_p1_win_rate_opt = total_p1_wins_opt / total_opt * 100
        print(
            f"    Player 1 wins: {total_p1_wins_opt}/{total_opt} ({total_p1_win_rate_opt:.1f}%)"
        )
        print(
            f"    Player 2 wins: {total_p2_wins_opt}/{total_opt} ({100 - total_p1_win_rate_opt:.1f}%)"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
