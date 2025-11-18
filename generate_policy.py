"""Generate optimal DP policy for Liar's Dice."""

from src.dp_solver import DPSolver
from src.opponent_policies import TruthTellingPolicy


def main():
    """Generate and save optimal policy."""
    num_dice = 5

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
