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
