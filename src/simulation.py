"""Game simulation engine."""

import random

from src.game_state import GameStatus, State, observable_from_state
from src.opponent_policies import TruthTellingPolicy
from src.policy import OptimalPolicy
from src.probability import p_valid
from src.rules import check_win_condition


class GameSimulation:
    """Simulates a game of Liar's Dice."""

    def __init__(self, num_dice_per_player: int = 5):
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

        if verbose:
            print("=" * 60)
            print("GAME START")
            print("=" * 60)
            print(f"Player 1 hand: {hand1}")
            print(f"Player 2 hand: {hand2}")
            print("Bid history: []")
            print("-" * 60)

        turn_number = 1  # Track turns from Player 1's perspective
        current_player = 1  # 1 = Player 1, 2 = Player 2

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
                        state, caller=1, last_bid=state.bid_history[-1]
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
                        state, caller=2, last_bid=state.bid_history[-1]
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
