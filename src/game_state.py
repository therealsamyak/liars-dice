"""Game state and data structures for Liar's Dice."""

from enum import Enum


class GameStatus(Enum):
    """Game status."""

    ACTIVE = "Active"
    GAME_OVER = "GameOver"


class Bid:
    """A bid: (value, amount)."""

    def __init__(self, value: int, amount: int):
        self.value = value
        self.amount = amount

    def __str__(self) -> str:
        return f"{self.amount} {self.value}s"


class State:
    """Complete game state: (H1, H2, D, Go)."""

    def __init__(
        self,
        hand1: list[int],
        hand2: list[int],
        bid_history: list[Bid],
        game_status: GameStatus,
    ):
        self.hand1 = hand1
        self.hand2 = hand2
        self.bid_history = bid_history
        self.game_status = game_status

    def get_last_bid(self) -> Bid | None:
        return self.bid_history[-1] if self.bid_history else None


class ObservableState:
    """Observable state for Player 1: (H1, D, Go)."""

    def __init__(
        self, hand1: list[int], bid_history: list[Bid], game_status: GameStatus
    ):
        self.hand1 = hand1
        self.bid_history = bid_history
        self.game_status = game_status


def observable_from_state(state: State) -> ObservableState:
    """Create observable state from full state."""
    return ObservableState(
        hand1=state.hand1.copy(),
        bid_history=state.bid_history.copy(),
        game_status=state.game_status,
    )
