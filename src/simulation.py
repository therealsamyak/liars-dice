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
        player1_policy=None,
        player2_policy=None,
    ):
        """
        Args:
            num_dice_per_player: Number of dice each player has
            player2_policy_type: "truth_telling", "optimal", or "rl" (uses same policy as Player 1)
            player1_policy: Optional pre-initialized policy for Player 1
            player2_policy: Optional pre-initialized policy for Player 2
        """
        self.num_dice_per_player = num_dice_per_player
        
        # Set player 1 policy
        if player1_policy is not None:
            self.player1_policy = player1_policy
        else:
            self.player1_policy = OptimalPolicy(num_dice_per_player)

        # Set player 2 policy
        if player2_policy is not None:
            self.player2_policy = player2_policy
        elif player2_policy_type == "optimal":
            self.player2_policy = OptimalPolicy(num_dice_per_player)
        elif player2_policy_type == "rl":
            # For RL, expect it to be provided via player2_policy parameter
            # Default to truth-telling if not provided
            self.player2_policy = TruthTellingPolicy(num_dice_per_player)
        else:
            self.player2_policy = TruthTellingPolicy(num_dice_per_player)

    def roll_dice(self, num_dice: int) -> list[int]:
        """Roll N dice."""
        return [random.randint(1, 6) for _ in range(num_dice)]

    def simulate_game(
        self, verbose: bool = True, starting_player: int | None = None
    ) -> tuple[int, State]:
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
                elif hasattr(self.player2_policy, 'get_action') and len(getattr(self.player2_policy.get_action, '__code__', type('', (), {'co_argcount': 2})).co_varnames) > 1:
                    # RL policy or other policy that takes ObservableState
                    obs2 = ObservableState(
                        hand1=state.hand2,  # Player 2's hand is their "hand1"
                        bid_history=state.bid_history.copy(),
                        game_status=state.game_status,
                    )
                    action = self.player2_policy.get_action(obs2)
                else:
                    # Player 2 uses truth-telling policy (hand, bid_history)
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
