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
        num_dice,
        opponent_policy=None,
        max_value=MAX_DICE_VALUE,
        max_amount=MAX_BID_AMOUNT,
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
