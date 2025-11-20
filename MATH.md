## 1 Game 2 Player Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- **Bidding rules (standard Liar's Dice):**
  - Each subsequent bid must increase either the quantity OR the face value
  - You can increase quantity while keeping same face value (e.g., "three 3s" -> "four 3s")
  - You can increase face value with ANY quantity (e.g., "three 3s" -> "one 4s" or "three 3s" -> "five 4s")
  - You CANNOT decrease face value
  - You CANNOT decrease quantity when keeping same face value
  - Examples:
    - "three 3s" -> "four 3s" [VALID] (increase quantity, same value)
    - "three 3s" -> "three 4s" [VALID] (increase value, same quantity)
    - "three 3s" -> "one 4s" [VALID] (increase value, any quantity allowed)
    - "three 3s" -> "five 4s" [VALID] (increase both)
    - "three 3s" -> "two 3s" [INVALID] (decrease quantity, same value)
    - "four 4s" -> "three 3s" [INVALID] (decrease value)
- At any point, instead of bidding, a player can call 'LIAR!', ending the game.
  - If the bid is invalid (COUNT < Amount), the caller wins
  - If the bid is valid (COUNT >= Amount), the bidder wins

### Math Formulations

$S = (H_1, H_2, D, G_o)$ - state space

- $H_1, H_2 \in \{1,2,3,4,5,6\}^N$ - dice hands for player 1 and player 2
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history $D_{last}$ - latest bid
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$O = (H_1, D, G_o)$ - observable space

- $H_1 \in \{1, 2, 3, 4, 5, 6\}^N$ - our player's dice hand
- $D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$ - bid history
- $G_o \in \{\text{Active}, \text{GameOver}\}$ - game status

$A \in \{B, C\}$ - Action set

- $B = (\text{DiceValue},\text{Amount})$ - new bid
- $C$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($G_o = \text{GameOver}$)

$T(s' \mid s, a)$ - transition probabilities

- If action is bid $B$: $H_1' = H_1, H_2' = H_2, D' = D \cup \{B\}, G_o' = \text{Active}$ (if $B \in \text{LEGAL}(D)$)
- If action is call LIAR $C$: $H_1' = H_1, H_2' = H_2, D' = D, G_o' = \text{GameOver}$

$T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, D' = D, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}$

Where:
$\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \text{ (any amount allowed when increasing value)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

**Bid Priority Order:** When generating legal bids, priority is:

1. First: increase face value (any quantity allowed, ordered by value then amount, lowest first)
2. Then: same face value, increase amount

- Hands $H_1, H_2$ remain constant throughout a single game. Only $D$ and $G_o$ change.

New observation: $\Omega(o \mid s', a) = (H_1', D', G_o')$

Reward: $R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$

Win Conditions: $WIN(s, caller)$ where $caller \in \{1, 2\}$ is the player who called LIAR:

$WIN(s, caller) = \begin{cases}
\text{P1} & \text{if } caller = 1, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if } caller = 1, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if } caller = 2, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if } caller = 2, G_o = \text{GameOver}, \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o = \text{Active}
\end{cases}$

Where:

- $\text{COUNT}(H, v)$ returns the number of dice in hand $H$ showing value $v$
- $D_{\text{last}}$ is the most recent bid
- If caller wins: bid was invalid (caller was correct to challenge)
- If bidder wins: bid was valid (caller was wrong to challenge)

$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$ - baseline prior over opponent's hand

- Any bid given from opponent is not valid proof of the hand state.

HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $b(H_2 \mid H_1, D_last)$

$D_last = (v*, a*)$ - value, amount

$P_{valid}(D_{last} \mid H_1) = P(COUNT(H_1 \cup H_2, v*) \ge a* | H_1)$

$c_1 = COUNT(H_1, v*)$, $c_2 = COUNT(H_2, v*)$ - number of dice in each hand with value $v*$
$k = a* - c_1$ - additional dice needed from opponent for bid to be valid

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k)$

Because we don't know opponent's hand / policy, $c_2 \sim Binomial(N, p = \frac{1}{6})$

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$

- If $k > N$, then $P_{valid} = 0$ - impossible bid
- If $k \le 0$, then $P_{valid} = 1$ - our hand satisfies bid

### Notes:

- Modeled from Player 1 perspective. Opponent actions handled by system internally.
