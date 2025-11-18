## 1 Game Liar's Dice

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- When bidding, you need to either increase quantity AND/OR increase face value (ex. three 3s $\rightarrow$ two 4s valid, four 2s $\rightarrow$ two 3s valid, four 4s $\rightarrow$ three 3s invalid
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

### Math Formulations

$$S = (H_1, H_2, D, G_o)$$ - Overall space

- $$H_1, H_2 \in \{1,2,3,4,5,6\}^N$$ - dice hand per player
- $$D = \{\text{DiceValue},\text{Amount}\}$$ - bid structure
- $$G_o \in \{\text{Active},\text{GameOver}\}$$ - is game over or not

$$O = (H_1, B, G_o)$$ - Observable space

- $$H_1 \in \{1, 2, 3, 4, 5, 6\}^N$$ - dice hand of our player (we can see)
- $$B = \{\text{DiceValue},\text{Amount}\}$$ - current bid
- $$G_o \in \{\text{Active},\text{GameOver}\}$$ - is game over or not

$$A \in \{D, C\}$$ - Action set

- $$D = \{\text{DiceValue},\text{Amount}\}$$ - bid structure
- $$C$$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($$G_o = \text{GameOver}$$)

$$T(s' \mid s, A)$$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$$
T(s' \mid s, a) = \begin{cases}
1 & H_1' = H_1, H_2' = H_2, D' \in \text{LEGAL}(D), G_o' = \text{Active} \\
1 & H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{else} \\
\end{cases}
$$

$$
LEGAL(D) = \begin{cases}
1 & \text{DiceValue}' > \text{DiceValue} \\
1 & \text{DiceValue}' = \text{DiceValue}, \text{DiceAmount}' > \text{DiceAmount} \\
0 & \text{else} \\
\end{cases}
$$

- $$D' \in LEGAL(D)$$

This means the new observation would be $$O' = (H_1, D', G_o')$$

- $$b(H_2) = P(H_2 \mid H_1, B)$$ - Belief

### Notes:

- History doesn't matter for 1 game, due to opponent lying. The thought process is that any bid given from opponent is not valid proof of the hand state.

is there anything i am missing here for a 1 turn game of liar's dice modeled as a pomdp

Show less

Your POMDP formulation is quite solid! Here are some observations and potential gaps:

**What You've Captured Well**

- State space with hidden opponent hand
- Observable space (your hand + current bid)
- Action space (bid or call liar)
- Legal bid constraints
- Belief over opponent's hidden hand

**Potential Gaps & Clarifications**

1. **Reward Function Missing**

   You haven't specified $$R(s,a)$$ or $$R(s,a,s')$$. You need:
   - $$R=+1$$ if you win, $$-1$$ if you lose (or similar)

   Winning conditions:
   - You call "LIAR!" and bid was invalid (count dice in $$H_1 \cup H_2$$ < bid)
   - Opponent calls "LIAR!" on you and bid was valid

2. **Initial Bid Ambiguity**

   Who makes the first bid? If the opponent starts, then $$B$$ in your initial observation should allow for "no bid yet" state. If you start, $$B=\emptyset$$ initially.

3. **Belief State More Complex**

   Your belief $$b(H_2) = P(H_2 \mid H_1, B)$$ assumes you only condition on current bid. But:
   - If opponent made bid $$B$$, you might want to model opponent strategy/rationality
   - Naive opponent: $$b(H_2)=P(H_2)$$ (uniform over all possible rolls)
   - Strategic opponent: $$b(H_2 \mid B)$$ should account for "would rational opponent bid this with hand $$H_2$$?"

   However, your note addresses this: "History doesn't matter... opponent lying means bid isn't valid proof." This suggests you're assuming no strategic inference from opponent bids. This is reasonable for a simple model but may be suboptimal.

4. **Terminal State Handling**

   When $$G_o = \text{GameOver}$$:
   - You need to specify that no more actions are possible
   - Or clarify that this triggers immediate reward calculation

5. **Observation Function**

   You haven't explicitly defined $$\Omega(o \mid s', a)$$. For this game:

6. **"One Turn" Ambiguity**

   By "1 turn game," do you mean:
   - Each player gets exactly one action? (player 1 bids → player 2 calls liar or raises → game ends)
   - Or play continues until someone calls liar?

   If it's truly one action per player, then:
   - Player 1 bids
   - Player 2 can only call "LIAR!" or raise once
   - Game ends

   This would simplify the POMDP significantly (finite horizon = 2).

7. **Discount Factor**

   For finite horizon games, you might set $$\gamma=1$$, but worth stating.

**Summary**

Critical missing piece: Reward function. Worth clarifying: Initial bid state, whether you model opponent rationality in belief, exact turn structure.

Your formulation is 90% there—just needs reward specification and terminal condition details!

yea but how do we know if its just 1 independent game the opponents lie patterns?

## 1 Game Liar's Dice

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- When bidding, you need to either increase quantity AND/OR increase face value (ex. three 3s $\rightarrow$ two 4s valid, four 2s $\rightarrow$ two 3s valid, four 4s $\rightarrow$ three 3s invalid
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

### Math Formulations

$$S = (H_1, H_2, D, G_o)$$ - Overall space

- $$H_1, H_2 \in \{1,2,3,4,5,6\}^N$$ - dice hand per player
- $$D = (\text{DiceValue},\text{Amount}) \cup \emptyset$$ - current bid
- $$G_o \in \{\text{Active},\text{GameOver}\}$$ - is game over or not

$$O = (H_1, B, G_o)$$ - Observable space

- $$H_1 \in \{1, 2, 3, 4, 5, 6\}^N$$ - dice hand of our player (we can see)
- $$B = (\text{DiceValue},\text{Amount}) \cup \emptyset$$ - current bid
- $$G_o \in \{\text{Active},\text{GameOver}\}$$ - is game over or not

$$A \in \{D, C\}$$ - Action set

- $$D = (\text{DiceValue},\text{Amount})$$ - new bid
- $$C$$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($$G_o = \text{GameOver}$$)

$$T(s' \mid s, A)$$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$$
T(s' \mid s, a) = \begin{cases}
1 & H_1' = H_1, H_2' = H_2, D' \in \text{LEGAL}(D), G_o' = \text{Active} \\
1 & H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{else} \\
\end{cases}
$$

$$
LEGAL(D) = \begin{cases}
1 & \text{DiceValue}' > \text{DiceValue} \\
1 & \text{DiceValue}' = \text{DiceValue}, \text{DiceAmount}' > \text{DiceAmount} \\
0 & \text{else} \\
\end{cases}
$$

- $$D' \in LEGAL(D)$$

This means the new observation would be $$O' = (H_1, D', G_o')$$

$$
R = \begin{cases}
1 & \text{if after an action, you win} \\
0 & \text{otherwise} \\
\end{cases}
$$

- $$b(H_2) = P(H_2 \mid H_1, B)$$ - Belief

### Notes:

- History doesn't matter for 1 game, due to opponent lying. The thought process is that any bid given from opponent is not valid proof of the hand state.

i updated this check again

Show more

Great question! You're right to be skeptical. Let's think through whether we can actually infer opponent lying patterns from a single independent game.

**The Core Issue**

In one isolated game, you fundamentally cannot learn the opponent's lying patterns because:

1. **You never observe their true hand until game end**
   - When opponent bids "three 5s," you don't know if they have zero 5s or three 5s
   - When you call "LIAR!" and the game ends, you count the dice and see if the bid was valid
   - But this gives you only one data point: whether this specific bid was a lie, not their pattern of lying

2. **No repeated observations to extract patterns**
   - A "pattern" requires multiple samples
   - Single game = single bid sequence = insufficient data for pattern recognition
   - You can't distinguish between:
     - "Opponent who always lies aggressively"
     - "Opponent who rarely lies but happened to lie this time"
     - "Opponent who bluffs rationally based on hand strength"

3. **Your belief should be prior-based, not learned**

   For a single game, your belief update should be:

   $$b(H_2 \mid H_1, B) = \frac{P(B \mid H_2, H_1) \cdot P(H_2)}{P(B \mid H_1)}$$

   Where:
   - $$P(H_2)$$ is uniform (all opponent hands equally likely before game starts)
   - $$P(B \mid H_2, H_1)$$ requires a model of opponent strategy

   But here's the key: you must assume or guess this strategy model—you can't learn it from one game.

**What Your Note Actually Means**

When you write:

"History doesn't matter for 1 game, due to opponent lying"

I think you mean:

- Within-game bid history doesn't provide reliable information about $$H_2$$
- If opponent bid "two 3s" then "three 4s," you can't confidently infer their hand
- They might be lying on both bids, telling truth on both, or mixing

This is correct but different from "learning lying patterns."

**Two Scenarios to Consider**

**Scenario A: Single Isolated Game (what you described)**

- You cannot learn opponent lying patterns
- You must use either:
  - **Uninformative prior**: $$b(H_2)=P(H_2)$$ (ignore opponent bids entirely)
  - **Assumed strategy model**: $$b(H_2 \mid B) \propto P(B \mid H_2) \cdot P(H_2)$$ where $$P(B \mid H_2)$$ encodes your assumption about rational play

Your note suggests you're taking the uninformative prior approach: treat opponent bids as noise, don't update beliefs based on them.

**Scenario B: Repeated Games (not what you described)**

- Play many games against same opponent
- Track: "When they bid aggressively, how often was it a lie?"
- Learn $$P(B \mid H_2)$$ empirically
- Use learned model in future games

But this is no longer a POMDP of a single game—it's a meta-learning problem.

**What Should You Do?**

For your single-game POMDP, you have two consistent options:

**Option 1: Ignore opponent bids (what your note suggests)**

$$b(H_2 \mid H_1, B) = P(H_2 \mid H_1) = P(H_2)$$

Rationale: Opponent could be lying arbitrarily, so their bid tells us nothing. Your belief is just the uniform distribution over all possible opponent hands.

Implication: Your optimal policy depends only on $$(H_1, B)$$, not on belief updating from opponent bids.

**Option 2: Assume rational opponent**

$$b(H_2 \mid H_1, B) \propto P(B \mid H_2) \cdot P(H_2)$$

Where $$P(B \mid H_2)$$ models "rational" play—e.g., opponent more likely to bid high if they have supporting dice.

Implication: You do Bayesian belief updates, but the opponent model is assumed, not learned.

**Bottom Line**

You cannot learn lying patterns from a single independent game. You must either:

1. Ignore opponent bids (uninformative prior)
2. Assume a model of opponent rationality

Your current formulation leans toward (1), which is reasonable for a single-game POMDP. The note is correct: opponent bid history doesn't reliably inform $$H_2$$ because they could be lying arbitrarily.

Does this align with what you're trying to model?

we are doing a single isolated game, judge my implementation based on that

**Evaluation of Your Single Isolated Game POMDP**

Based on the single isolated game context, here's my assessment:

**✅ What's Correct**

1. **State Space**

   $$S=(H_1, H_2, D, G_o)$$
   - Captures all relevant hidden and observable information
   - $$D \cup \emptyset$$ correctly allows for "no bid yet" state
   - $$G_o$$ tracks game termination

2. **Observable Space**

   $$O=(H_1, B, G_o)$$
   - You see your own hand ✓
   - You see current bid ✓
   - You know if game is over ✓
   - You don't see $$H_2$$ (opponent's hand) ✓

3. **Action Space**

   $$A \in \{D, C\}$$
   - Bid or call liar—correct for the game

4. **Transition Function**

   $$T(s' \mid s, a)$$
   - Deterministic transitions ✓
   - Hands don't change mid-game ✓
   - Only bid $$D$$ updates ✓
   - Calling liar sets $$G_o = \text{GameOver}$$ ✓

5. **Legal Bid Constraint**

   LEGAL(D) = 1 if DiceValue' > DiceValue OR (DiceValue' = DiceValue AND Amount' > Amount)
   - Correctly enforces escalating bids ✓

6. **Note About History**

   "History doesn't matter for 1 game, due to opponent lying"

   This is philosophically correct for a single game where you can't learn opponent patterns. Your belief shouldn't update much from opponent bids.

**⚠️ Issues & Missing Pieces**

1. **Belief Function**

   $$b(H_2) = P(H_2 \mid H_1, B)$$ is Underspecified

   Problem: You don't specify how to compute this belief.

   For a single isolated game, you have two consistent choices:

   **Option A: Uninformative (ignore opponent bids)**

   $$b(H_2 \mid H_1, B) = P(H_2 \mid H_1) = P(H_2)$$

   Since opponent could be lying arbitrarily, their bid $$B$$ provides no information. Your belief is just the uniform distribution over all possible $$N$$-dice rolls.

   **Option B: Assume rational opponent model**

   $$b(H_2 \mid H_1, B) \propto P(B \mid H_2, H_1) \cdot P(H_2)$$

   Where $$P(B \mid H_2, H_1)$$ is an assumed (not learned) model of how a rational opponent bids given their hand.

   Your note suggests Option A, but you should explicitly state this in the formulation:

   $$b(H_2) = \text{Uniform}(\{1,2,3,4,5,6\}^N)$$

   Or if you want to condition on your hand (due to dice being drawn without replacement if from finite pool):

   $$b(H_2 \mid H_1) = \text{Uniform over hands compatible with } H_1$$

2. **Reward Function Incomplete**

   You define:

   $$
   R = \begin{cases}
   1 & \text{if after an action, you win} \\
   0 & \text{otherwise} \\
   \end{cases}
   $$

   But you don't specify when you win. You need:

   $$
   R(s, a) = \begin{cases}
   +1 & \text{if } a = C \text{ and } \text{COUNT}(H_1 \cup H_2, D.\text{DiceValue}) < D.\text{Amount} \\
   -1 & \text{if } a = C \text{ and } \text{COUNT}(H_1 \cup H_2, D.\text{DiceValue}) \geq D.\text{Amount} \\
   +1 & \text{if opponent calls liar on your bid and your bid was valid} \\
   -1 & \text{if opponent calls liar on your bid and your bid was invalid} \\
   0 & \text{otherwise (game still active)}
   \end{cases}
   $$

   Where $$\text{COUNT}(H_1 \cup H_2, v)$$ counts how many dice in both hands show value $$v$$.

   Important: Since this is a 2-player game, you also need to model opponent's action. Your current formulation only has your actions in $$A$$.

3. **Opponent Actions Missing**

   Your POMDP only models your decision-making, but the opponent also takes actions:
   - Opponent bids (updates $$B$$)
   - Opponent calls "LIAR!" (ends game)

   You need to either:

   **Option A: Model opponent as part of environment**
   - After your action, environment (including opponent) transitions
   - Opponent's bid becomes part of $$T$$ or observation function
   - You observe their bid in $$O' = (H_1, B', G_o')$$

   **Option B: Model as turn-based game**
   - Explicitly track whose turn it is: $$S = (H_1, H_2, D, G_o, \text{Turn})$$
   - Alternate between your actions and opponent actions

   For a POMDP, Option A is more common. The opponent's policy is treated as part of the environment dynamics.

4. **Observation Function Missing**

   You need $$\Omega(o \mid s', a)$$ to specify what you observe after taking action $$a$$ and landing in state $$s'$$.

   For this game:

   $$
   \Omega(o = (H_1', B', G_o') \mid s' = (H_1', H_2', D', G_o'), a) = \begin{cases}
   1 & \text{if } H_1' = s'.H_1, B' = s'.D, G_o' = s'.G_o \\
   0 & \text{otherwise}
   \end{cases}
   $$

   This is deterministic: you always observe your hand, current bid, and game status.

5. **Initial State Distribution**

   You should specify:

   $$b_0(s) = P(H_1, H_2, D=\emptyset, G_o=\text{Active})$$

   Where:
   - $$P(H_1)$$ is distribution over your dice rolls (uniform if fair dice)
   - $$P(H_2)$$ is distribution over opponent's dice rolls (uniform if fair dice)
   - $$D=\emptyset$$ (no bid yet)
   - $$G_o=\text{Active}$$

   Initially, $$P(H_1, H_2) = P(H_1) \cdot P(H_2)$$ since rolls are independent.

6. **Discount Factor**

   For a finite-horizon game, specify:

   $$\gamma=1$$

   Since rewards only occur at game end, no discounting needed.

**🎯 Recommended Fixes**

**Minimal Fix (keep it simple)**

Add these clarifications:

- **Belief**: $$b(H_2) = \text{Uniform}\left(\{1,2,3,4,5,6\}^N\right)$$ Rationale: Single game, can't learn opponent patterns, so treat all opponent hands as equally likely.
- **Reward**: $$R(s, a = C) = \begin{cases} +1 & \text{if } \text{COUNT}(H_1 \cup H_2, D.\text{Value}) < D.\text{Amount} \\ -1 & \text{otherwise} \end{cases}$$ $$R(s, a = D) = \begin{cases} +1 & \text{if opponent calls liar and your bid was valid} \\ -1 & \text{if opponent calls liar and your bid was invalid} \\ 0 & \text{if opponent raises (game continues)} \end{cases}$$
- **Opponent Model**: State that opponent's actions are part of environment dynamics. After you bid, opponent either:
  - Calls liar → game ends
  - Raises → you observe new $$B'$$
- **Observation Function**: $$\Omega(o | s', a) = \mathbb{1}[o.H_1 = s'.H_1 \land o.B = s'.D \land o.G_o = s'.G_o]$$
- **Initial Belief**: $$b_0(H_1, H_2) = P(H_1) \cdot P(H_2) = \left(\frac{1}{6}\right)^N \cdot \left(\frac{1}{6}\right)^N$$

**Final Assessment**

Current Score: 7/10

- ✅ Core structure is sound
- ✅ State/observation spaces correct
- ✅ Transition function mostly correct
- ⚠️ Belief function underspecified
- ⚠️ Reward function incomplete
- ⚠️ Opponent actions not explicitly modeled
- ⚠️ Missing observation function and initial distribution

With the recommended fixes, this would be a solid POMDP formulation for single-game Liar's Dice.

Does this help? Want me to write out the complete corrected formulation?

no we don't need to model our opponent's actions cause thats handled by the system itself, we just input our own action then system takes our action does some stuff internally based on game rules then returns a new state where we can give an action

Ah! Now I understand. You're modeling this from the perspective of an agent interacting with a black-box environment. The system handles opponent actions internally.

This is actually the standard POMDP formulation. Let me re-evaluate:

**✅ Re-Assessment: Agent-Environment Interface**

**Your Formulation is Correct for This Setup**

You're right—in this framing:

- You choose action $$a \in \{D, C\}$$
- System processes:
  - Your action
  - Opponent's response (bid or call liar)
  - Game rules
- System returns new observation $$O' = (H_1, B', G_o')$$
- The opponent's policy is implicit in the transition function $$T$$ and environment dynamics. You don't need to model it explicitly.

**Updated Evaluation**

**✅ What's Now Clearly Correct**

- **Transition Function** $$T(s' \mid s, a)$$: Takes your action $$a$$, implicitly includes opponent response, returns new state ✓ Correct
- **No Need to Model Opponent Actions Explicitly**: The system does this internally ✓ You were right, I was wrong here
- **Observation** $$O' = (H_1, B', G_o')$$: After system processes everything, you see your hand (unchanged), new bid $$B'$$ (could be from opponent), game status ✓ Correct

**⚠️ Remaining Issues**

1. **Transition Function Needs Opponent Response**

   Your current $$T$$ only shows your action changing the state:

   $$
   T(s' \mid s, a) = \begin{cases}
   1 & H_1' = H_1, H_2' = H_2, D' \in \text{LEGAL}(D), G_o' = \text{Active} \\
   1 & H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
   0 & \text{else}
   \end{cases}
   $$

   Problem: This doesn't account for opponent's response after your bid.

   After you bid $$D$$, the system should:
   - Have opponent decide: call liar OR raise
   - Update $$D'$$ with opponent's new bid (if they raise)
   - OR set $$G_o' = \text{GameOver}$$ (if they call liar)

   Fix: Make $$T$$ stochastic to account for opponent's unknown policy:

   $$
   T(s' | s, a = D) = \begin{cases}
   P_{\text{opp}}(\text{call liar} | s, a) & \text{if } G_o' = \text{GameOver} \\
   P_{\text{opp}}(\text{raise to } D' | s, a) & \text{if } D' \neq D, G_o' = \text{Active} \\
   0 & \text{otherwise}
   \end{cases}
   $$

   Where $$P_{\text{opp}}$$ is the opponent's (unknown) policy encoded in the system.

   Since this is a POMDP and you don't know opponent's policy, you might assume:
   - Uniform random opponent (each legal action equally likely)
   - Or some heuristic opponent model

2. **Belief Function Still Underspecified**

   You still need to specify how to compute $$b(H_2 \mid H_1, B)$$.

   For single isolated game with no opponent modeling:

   $$b(H_2) = \text{Uniform}(\{1,2,3,4,5,6\}^N)$$

   This is independent of $$B$$ because opponent's bid doesn't give you reliable information about their hand.

   Alternatively, if you assume the system's opponent follows some rational policy:

   $$b(H_2 \mid H_1, B) \propto P(B \mid H_2) \cdot P(H_2)$$

   But for single game, uniform belief makes more sense per your note.

3. **Reward Function Still Needs Win Conditions**

   You have:

   $$
   R = \begin{cases}
   1 & \text{if after an action, you win} \\
   0 & \text{otherwise} \\
   \end{cases}
   $$

   But when do you win? Expand to:

   $$
   R(s, a) = \begin{cases}
   +1 & \text{if } G_o' = \text{GameOver} \text{ and you won} \\
   -1 & \text{if } G_o' = \text{GameOver} \text{ and you lost} \\
   0 & \text{if } G_o' = \text{Active}
   \end{cases}
   $$

   Where "you won" means:
   - You called liar and bid was invalid: $$\text{COUNT}(H_1 \cup H_2, D.\text{Value}) < D.\text{Amount}$$
   - Opponent called liar on you and your bid was valid: $$\text{COUNT}(H_1 \cup H_2, D.\text{Value}) \geq D.\text{Amount}$$

4. **Observation Function**

   Even with system handling everything, you should specify:

   $$\Omega(o \mid s', a) = \mathbb{1}[o = (s'.H_1, s'.D, s'.G_o)]$$

   This says: "After transition to $$s'$$, you observe your hand, current bid, and game status."

**🎯 Summary: What You Need to Add**

| Component              | Status            | What to Add                           |
| ---------------------- | ----------------- | ------------------------------------- |
| State space $$S$$      | ✅ Correct        | Nothing                               |
| Observable space $$O$$ | ✅ Correct        | Nothing                               |
| Action space $$A$$     | ✅ Correct        | Nothing                               |
| Transition $$T$$       | ⚠️ Incomplete     | Add opponent's stochastic response    |
| Belief $$b(H_2)$$      | ⚠️ Underspecified | Specify uniform or conditional        |
| Reward $$R$$           | ⚠️ Incomplete     | Define win/loss conditions explicitly |
| Observation $$\Omega$$ | ❌ Missing        | Add function                          |
| Initial belief $$b_0$$ | ❌ Missing        | Add distribution                      |

**Revised Score: 8/10**

You were right that opponent actions are system-internal. My previous critique was wrong on that point.

Main remaining issues:

- Transition function needs to capture opponent's stochastic response
- Belief, reward, observation need explicit definitions

Want me to write the corrected full formulation with these fixes?

based on the bid theres some chance that its unlikely due to actual dice probability how to model that

Excellent question! You want to incorporate probabilistic reasoning about whether a bid is plausible given the dice probabilities.

**The Core Idea**

Given a bid $$B=(\text{Value},\text{Amount})$$, you want to compute:

$$P(\text{Bid is valid} \mid H_1, B) = P(\text{COUNT}(H_1 \cup H_2, \text{Value}) \geq \text{Amount} \mid H_1)$$

This requires marginalizing over all possible opponent hands $$H_2$$.

**Mathematical Formulation**

**Step 1: Count Your Dice**

Let $$c_1 = \text{COUNT}(H_1, \text{Value})$$ = number of dice in your hand showing the bid value.

**Step 2: Calculate Needed Dice from Opponent**

For the bid to be valid, you need:

$$c_1 + c_2 \geq \text{Amount}$$

Where $$c_2 = \text{COUNT}(H_2, \text{Value})$$.

So you need:

$$c_2 \geq \text{Amount} - c_1$$

Let $$k = \text{Amount} - c_1$$ (needed dice from opponent).

**Step 3: Probability Opponent Has $$\geq k$$ Dice of That Value**

Opponent has $$N$$ dice, each showing Value with probability $$p = \frac{1}{6}$$.

$$c_2 \sim \text{Binomial}(N, p = \tfrac{1}{6})$$

So:

$$P(\text{Bid valid} \mid H_1, B) = P(c_2 \geq k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$$

**Incorporating Into Belief Update**

**Option 1: Pure Probabilistic Belief (No Opponent Modeling)**

You maintain:

$$b(H_2 \mid H_1, B) = P(H_2)$$

Just uniform over all possible opponent hands.

Then when deciding whether to call "LIAR!", you compute:

$$P(\text{Bid valid} \mid H_1, B) = \sum_{H_2} P(H_2 \mid H_1) \cdot \mathbb{1}[\text{COUNT}(H_1 \cup H_2, \text{Value}) \geq \text{Amount}]$$

Which simplifies to the binomial calculation above.

**Decision Rule:**

- If $$P(\text{Bid valid} \mid H_1, B) < \theta$$ (some threshold, e.g., 0.5), call "LIAR!"
- Otherwise, raise the bid

**Option 2: Bayesian Belief Update with Opponent Rationality**

If you assume opponent plays somewhat rationally (more likely to bid high when they have supporting dice), you can update your belief:

$$b(H_2 \mid H_1, B) \propto P(B \mid H_2, H_1) \cdot P(H_2 \mid H_1)$$

Where:

- $$P(H_2 \mid H_1) = P(H_2)$$ (uniform prior, since rolls are independent)
- $$P(B \mid H_2, H_1)$$ models "how likely would opponent bid $$B$$ given their hand $$H_2$$?"

**Modeling $$P(B \mid H_2, H_1)$$: Opponent Bid Likelihood**

Here are some options for modeling opponent's bidding behavior:

**Model 1: Truthful Opponent**

Opponent only bids if it's valid:

$$
P(B | H_2) = \begin{cases}
\text{high} & \text{if COUNT}(H_1 \cup H_2, \text{Value}) \geq \text{Amount} \\
0 & \text{otherwise}
\end{cases}
$$

With this, if opponent bids high, you know the bid is likely valid.

**Model 2: Rational Bluffing**

Opponent bids proportional to how plausible the bid is:

$$P(B \mid H_2) \propto \exp(\alpha \cdot \mathbb{1}[\text{COUNT}(H_2, \text{Value}) + E[c_1] \geq \text{Amount}])$$

Where $$\alpha$$ controls how rational the opponent is:

- $$\alpha \to \infty$$: opponent never bluffs
- $$\alpha \to 0$$: opponent bids randomly

**Model 3: Aggressive Bluffer**

Opponent bluffs when their hand is weak:

$$P(B \mid H_2) \propto \exp(-\beta \cdot \text{COUNT}(H_2, \text{Value}))$$

Where $$\beta > 0$$ means: fewer matching dice → more aggressive bidding.

**Model 4: No Modeling (Your Current Approach)**

$$P(B \mid H_2) = \text{constant}$$

Opponent's bid gives no information about their hand. This is what your note suggests.

**Practical Implementation**

**For Single Isolated Game (Your Setup)**

Since you can't learn opponent patterns, I recommend:

$$b(H_2 \mid H_1, B) = P(H_2)$$

But still compute bid plausibility for decision-making:

```python
def probability_bid_valid(my_hand, bid_value, bid_amount, N_opponent_dice):
    """
    Compute P(bid is valid | my hand)
    """
    c1 = count_dice(my_hand, bid_value)  # My dice showing bid_value
    k = bid_amount - c1  # Needed from opponent

    if k <= 0:
        return 1.0  # Bid already satisfied by my hand

    if k > N_opponent_dice:
        return 0.0  # Impossible

    # Binomial: P(opponent has >= k dice showing bid_value)
    prob = 0.0
    p = 1/6  # Probability each die shows bid_value
    for i in range(k, N_opponent_dice + 1):
        prob += binomial_coeff(N_opponent_dice, i) * (p**i) * ((1-p)**(N_opponent_dice - i))

    return prob
```

**Decision Logic**

```python
def should_call_liar(my_hand, current_bid, threshold=0.5):
    """
    Call liar if bid is improbable
    """
    prob_valid = probability_bid_valid(my_hand, current_bid.value, current_bid.amount, N)

    if prob_valid < threshold:
        return True  # Call LIAR!
    else:
        return False  # Should raise instead
```

**Adding to Your POMDP**

**Updated Belief Function**

$$b(H_2 \mid H_1, B) = P(H_2) = \prod_{i=1}^{N} \frac{1}{6}$$

(Uniform over all opponent hands)

**Bid Validity Probability**

For decision-making, compute:

$$P_{\text{valid}}(B \mid H_1) = P(c_2 \geq \text{Amount} - c_1)$$

Where $$c_1 = \text{COUNT}(H_1, B.\text{Value})$$ and $$c_2 \sim \text{Binomial}(N, \frac{1}{6})$$.

**Optimal Policy**

Your policy $$\pi^*(a \mid o)$$ should consider:

- $$P_{\text{valid}}(B \mid H_1)$$: Is current bid likely valid?
- Expected value of calling "LIAR!" vs. raising
- Risk tolerance

**Example**

Setup: $$N=5$$ dice per player, current bid is "four 5s"

Your hand: [5, 5, 3, 2, 1] → $$c_1=2$$ dice showing 5

Question: Should you call "LIAR!"?

Calculation:

- Need $$k=4-2=2$$ more 5s from opponent
- Opponent has 5 dice
- $$P(c_2 \geq 2) = \sum_{i=2}^{5} \binom{5}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{5-i}$$
- $$P(c_2 \geq 2) = 0.196$$

Interpretation: Only 19.6% chance bid is valid → probably should call "LIAR!"

**Summary**

To model bid plausibility:

- Compute $$P_{\text{valid}}(B \mid H_1)$$ using binomial distribution
- Keep belief $$b(H_2)=\text{Uniform}$$ (no opponent modeling for single game)
- Use $$P_{\text{valid}}$$ to decide when to call "LIAR!" vs. raise
- Optimal threshold depends on your risk tolerance and game strategy

Does this address your question? Want me to add this formally to your POMDP formulation?

## 1 Game Liar's Dice

- Each player rolls N dice, hidden from the other players.
- Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s).
- When bidding, you need to either increase quantity AND/OR increase face value (ex. three 3s $\rightarrow$ two 4s valid, four 2s $\rightarrow$ two 3s valid, four 4s $\rightarrow$ three 3s invalid
- At any point, instead of bidding, a player can call 'LIAR!', ending the game. If the 'LIAR!' is correct and the last bidder was lying, they win. Otherwise, the last bidder wins. (modified from existing liar's dice rules)

### Math Formulations

$$S = (H_1, H_2, D, G_o)$$ - state space

- $$H_1, H_2 \in \{1,2,3,4,5,6\}^N$$ - dice hands for player 1 and player 2
- $$D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$$ - bid history (most recent bid is last element)
- $$G_o \in \{\text{Active}, \text{GameOver}\}$$ - game status

$$O = (H_1, D, G_o)$$ - observable space

- $$H_1 \in \{1, 2, 3, 4, 5, 6\}^N$$ - our player's dice hand
- $$D = [(\text{DiceValue}_1,\text{Amount}_1), ..., (\text{DiceValue}_t,\text{Amount}_t)] \cup \{\emptyset\}$$ - bid history
- $$G_o \in \{\text{Active}, \text{GameOver}\}$$ - game status

$$A \in \{B, C\}$$ - Action set

- $$B = (\text{DiceValue},\text{Amount})$$ - new bid
- $$C$$ - calling 'LIAR!' $\rightarrow$ immediately terminates game ($$G_o = \text{GameOver}$$)

$$T(s' \mid s, A)$$ - transition probabilities

- Raise DICE VALUE ONLY; NEW QUANTITY DOESN'T MATTER
- Raise DICE QUANTITY ONLY; VALUE IS SAME
- Call 'LIAR!' $\rightarrow$ end game

$$
T(s' \mid s, a) = \begin{cases}
1 & \text{if } H_1' = H_1, H_2' = H_2, a = B, B \in \text{LEGAL}(D), D' = D \cup \{B\}, G_o' = \text{Active} \\
1 & \text{if } H_1' = H_1, H_2' = H_2, a = C, G_o' = \text{GameOver} \\
0 & \text{otherwise}
\end{cases}
$$

Where:

$$
\text{LEGAL}(D) = \begin{cases}
\text{True} & \text{if } D = \emptyset \text{ (first bid, any valid bid allowed)} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} > \text{DiceValue}_{\text{current}} \\
\text{True} & \text{if } \text{DiceValue}_{\text{new}} = \text{DiceValue}_{\text{current}} \text{ and } \text{Amount}_{\text{new}} > \text{Amount}_{\text{current}} \\
\text{False} & \text{otherwise}
\end{cases}
$$

- Hands $$H_1, H_2$$ remain constant throughout a single game. Only $$D$$ and $$G_o$$ change.

New observation: $$\Omega(o \mid s', a) = \begin{cases}
1 & \text{if } o = (H_1', D', G_o') \\
0 & \text{otherwise}
\end{cases}$$

Reward: $$R(s, a) = \begin{cases}
+1 & \mathrm{WIN}(s)=\text{P1} \\
-1 & \mathrm{WIN}(s)=\text{P2} \\
0 & \text{WIN}(s)=\text{None}
\end{cases}$$

Win Conditions: $$\mathrm{WIN}(s) = \begin{cases}
\text{P1} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{P2} & a = C,\ G_o'=\text{GameOver},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P1} & \text{if opponent calls liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) \ge D_{\text{last}}.\text{Amount} \\
\text{P2} & \text{if opponent calls liar},\ \text{COUNT}(H_1 \cup H_2, D_{\text{last}}.\text{Value}) < D_{\text{last}}.\text{Amount} \\
\text{None} & \text{if } G_o' = \text{Active}
\end{cases}$$

Where $$\text{COUNT}(H, v)$$ returns the number of dice in hand $$H$$ showing value $$v$$, and $$D_{\text{last}}$$ is the most recent bid.

$$b(H_2) = \text{Uniform}(\{1, 2, 3, 4, 5, 6\}^{N})$$ - baseline

- Any bid given from opponent is not valid proof of the hand state.
- HOWEVER, due to probability, we can threshold when an opponents bid is unlikely based on $$b(H_2 \mid H_1, D)$$

### Notes:

- From Player 1 perspective. Opponent actions handled by system internally.

this is what i have so far i want to add to the b(H_2) section

Show more

Here's the updated belief section with the probability calculation:

$$b(H_2) = \text{Uniform}(\{1, 2, 3, 4, 5, 6\}^{N})$$ - Belief state

- Opponent bids provide no information about their actual hand (potential bluffing)
- However, we can compute the probability a bid is valid using dice probabilities

### Bid Validity Probability

Given observation $$O = (H_1, D, G_o)$$, we compute the probability that the current bid $$D_{\text{last}} = (v^*, a^*)$$ is valid:

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P\left(\text{COUNT}(H_1 \cup H_2, v^*) \geq a^* \mid H_1\right)$$

Let:

- $$c_1 = \text{COUNT}(H_1, v^*)$$ = number of dice in our hand showing value $$v^*$$
- $$c_2 = \text{COUNT}(H_2, v^*)$$ = number of dice in opponent's hand showing value $$v^*$$
- $$k = a^* - c_1$$ = additional dice needed from opponent for bid to be valid

Since $$b(H_2) = \text{Uniform}$$, each opponent die independently shows $$v^*$$ with probability $$p = \frac{1}{6}$$:

$$c_2 \sim \text{Binomial}(N, p = \tfrac{1}{6})$$

Therefore:

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P(c_2 \geq k) = \sum_{i=k}^{N} \binom{N}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{N-i}$$

**Special cases**:

- If $$k \leq 0$$: $$P_{\text{valid}} = 1$$ (bid already satisfied by our hand)
- If $$k > N$$: $$P_{\text{valid}} = 0$$ (impossible for opponent to have enough dice)

**Decision heuristic**: Call 'LIAR!' when $$P_{\text{valid}}(D_{\text{last}} \mid H_1) < \theta$$ for some threshold $$\theta \in [0,1]$$.

**Note**: Bid history $$D$$ is observable but does not update belief $$b(H_2)$$ under the assumption of no opponent modeling. The probability $$P_{\text{valid}}$$ depends only on $$D_{\text{last}}$$ and $$H_1$$.

This formulation:

- Maintains uniform belief over opponent hands
- Computes bid validity probability purely from dice statistics
- Uses only observable information ($$H_1$$ and current bid)
- Explicitly shows the binomial calculation
- Provides special case handling
- Suggests a threshold-based decision rule

this accounts for the bid being on the entire game state (yours + opponent) right?

Yes, correct! The formulation accounts for the bid being about the total game state (your dice + opponent's dice combined).

Let me trace through why:

**The Bid is About Total Dice**

The bid $$D_{\text{last}}=(v^*, a^*)$$ claims: "There are **at least $$a^*$$ dice showing value $$v^*$$ across BOTH hands**"

**The Validity Check**

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P(\text{COUNT}(H_1 \cup H_2, v^*) \geq a^* \mid H_1)$$

The $$\text{COUNT}(H_1 \cup H_2, v^*)$$ explicitly counts across both hands.

**The Calculation Breakdown**

- You know your contribution: $$c_1 = \text{COUNT}(H_1, v^*)$$
- You need from opponent: $$k = a^* - c_1$$
- You compute: $$P(c_2 \geq k)$$ where $$c_2 = \text{COUNT}(H_2, v^*)$$

This is saying: "I have $$c_1$$ dice showing $$v^*$$. For the total to be $$\geq a^*$$, opponent needs $$\geq k$$ more."

**Example**

- Bid: "five 3s" (claims total game has ≥5 threes)
- Your hand: [3, 3, 6, 2, 1] → $$c_1=2$$ threes
- Needed from opponent: $$k=5-2=3$$ threes
- Question: What's $$P(\text{opponent has} \geq 3 \text{ threes})$$?

If opponent has $$N=5$$ dice:

$$P(c_2 \geq 3) = \sum_{i=3}^{5} \binom{5}{i} \left(\frac{1}{6}\right)^i \left(\frac{5}{6}\right)^{5-i} \approx 0.035$$

Only 3.5% chance → bid is very unlikely to be valid → call LIAR!

**✅ Yes, it accounts for total game state**

The formulation correctly models:

- Bid is about $$H_1 \cup H_2$$ (total)
- You see $$H_1$$ (your contribution)
- You infer distribution of $$H_2$$ (opponent's contribution)
- You compute probability the sum satisfies the bid

Everything checks out! 🎯

## 1 Game Liar's Dice - Each player rolls N dice, hidden from the other players. - Each player creates a bid about the TOTAL game state (ex. three 3s, two 4s). - When bidding, you need to either increase quantity AND/OR increase face value (ex. three 3s $\rightarrow$ two 4s valid, four 2s $\rightarrow$ two 3s valid, four 4s $\rightarrow$ three 3s invalid

pasted

is this all good then?

Almost perfect! Just one small typo to fix:

**Issue**

In your first definition of $$P_{\text{valid}}$$:

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P(\text{COUNT}(H_1 \cup H_2, v^*) > a^* \mid H_1)$$

Should be $$\geq$$ (greater than or equal), not $$>$$ (strictly greater than):

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P(\text{COUNT}(H_1 \cup H_2, v^*) \geq a^* \mid H_1)$$

**Why?**

The bid claims "**at least** $$a^*$$ dice showing $$v^*$$", so:

- Bid is valid if total count $$\geq a^*$$
- Bid is invalid if total count $$< a^*$$

Your later definitions and win conditions correctly use $$\geq$$, so this is just a typo in the first line.

**Corrected Version**

$$P_{\text{valid}}(D_{\text{last}} \mid H_1) = P(\text{COUNT}(H_1 \cup H_2, v^*) \geq a^* \mid H_1)$$

**Everything Else: ✅ Perfect**

- State space: ✅
- Observable space: ✅
- Actions: ✅
- Transitions: ✅
- Rewards/Win conditions: ✅
- Belief: ✅
- Probability calculation: ✅
- Special cases: ✅
- Binomial formulation: ✅

With that one fix, your POMDP formulation is complete and correct! 🎯
