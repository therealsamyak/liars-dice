"""Main entry point for Liar's Dice simulation."""

from src.simulation import GameSimulation


def main():
    """Run the Liar's Dice simulation with optimal policy."""
    print("=== Liar's Dice Simulation ===\n")

    # Create simulation with default 5 dice per player
    simulation = GameSimulation(num_dice_per_player=5)

    # Run a single game
    winner, final_state = simulation.simulate_game(verbose=True)

    print("\n=== Game Over ===")
    print(f"Final winner: Player {winner}")
    print(f"Total bids made: {len(final_state.bid_history)}")


if __name__ == "__main__":
    main()
