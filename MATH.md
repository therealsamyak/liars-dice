## 1 Game Liar's Dice

Authors: Samyak Kakatur, Alexander Chang

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- When bidding, you need to either increase quantity AND/OR increase face value (ex. three 3s $\rightarrow$ two 4s valid, four 2s $\rightarrow$ two 3s valid, four 4s $\rightarrow$ three 3s invalid
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

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

$T(s' \mid s, A)$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}$

Where:
$\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}$

- Hands $H_1, H_2$ remain constant throughout a single game. Only $D$ and $G_o$ change.

New observation: $\Omega(o \mid s', a) = (H_1', D', G_o')$
<br>

Reward: $R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$

Win Conditions: $WIN(s) = \begin{cases}
\text{P1} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if opponent\ calls\ liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o' = \text{Active}
\end{cases}$

Where $\text{COUNT}(H, v)$ returns the number of dice in hand $H$ showing value $v$, and $D_{\text{last}}$ is the most recent bid.

$b(H_2) = Uniform(\{1, 2, 3, 4, 5, 6\}^{N})$ - baseline

- Any bid given from opponent is not valid proof of the hand state.

HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $b(H_2 \mid H_1, D_last)$

$D_last = (v*, a*)$ - value, amount

$P_{valid}(D_{last} \mid H_1) = P(COUNT(H_1 \cup H_2, v*) \ge a* | H_1)$

$c_1 = COUNT(H_1, v*)$, $c_2 = COUNT(H_2, v*)$ - number of dice in each hand with value $v*$
$k = a* - c_1$ - additional dice needed from opponent for bid to be valid

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k)$

Because we don't know opponent's hand / policy, $c_2 ~ Binomial(N, p = \frac{1}{6}$

$P_{valid}(D_{last} \mid H_1) = P(c_2 \ge k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$

- If $k > N$, then $P_{valid} = 0$ - impossible bid
- If $k \le 0$, then $P_{valid} = 1$ - our hand satisfies bid

### Notes:

- Modeled from Player 1 perspective. Opponent actions handled by system internally.
