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
