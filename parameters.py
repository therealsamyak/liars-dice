"""System parameters for Liar's Dice simulation.

All system-wide constants are defined here and imported by other modules.
"""

# Dice configuration
NUM_DICE_PER_PLAYER = 2
NUM_PLAYERS = 2  # Number of players in the game

# Bid space configuration
MAX_DICE_VALUE = 6  # Maximum face value on a die (1-6)
MAX_BID_AMOUNT = (
    NUM_DICE_PER_PLAYER * NUM_PLAYERS
)  # Maximum amount in a bid (all dice same value)

# Derived constants
MAX_BID_POSITION = MAX_DICE_VALUE * MAX_BID_AMOUNT  # Total number of possible bids
