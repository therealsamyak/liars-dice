# Create MATH2.md for Liar's Dice

_Exported on 11/25/2025 at 11:23:56 PST from Cursor (2.1.26)_

---

**User**

add a new file MATH2.md with its h2 section called Liar's Dice w/ Reinforcement Learning, and fill it out based on the MATH1.md and the following GENERAL REQUIREMENTS PROVIDED, WE ARE TRYING TO USE THEM WITH OUR LIAR'S DICE:

@MATH2.md (9-34)

if something is mentioned in MATH.md don't duplciate it just mention that its in MATH.md

---

**Cursor**

Reviewing MATH.md and MATH2.md to understand the structure and avoid duplication.

Filling out MATH2.md with the reinforcement learning section, referencing MATH.md to avoid duplication.

Filling out MATH2.md with the reinforcement learning section:

```markdown
## Liar's Dice w/ Reinforcement Learning

This section extends the Liar's Dice formulation from MATH.md to support reinforcement learning approaches. The core mathematical formulation (state space, action space, system dynamics, sensor model, reward) remains as defined in MATH.md. Here we focus on the RL-specific extensions and implementations.

### Mathematical Formulation

The precise mathematical formulation is defined in MATH.md (lines 24-119). Key components:

- **State space** $S$: Defined in MATH.md (line 26)
- **Action/input space** $A$: Defined in MATH.md (line 38)
- **System dynamics** $T(s' \mid s, a)$: Defined in MATH.md (lines 43-60)
- **Sensor model** $\Omega(o \mid s', a)$: Defined in MATH.md (line 69)
- **Reward function** $R(s, a)$: Defined in MATH.md (lines 71-85)

### Adversary Policy

The opponent (Player 2) uses a heuristic policy that is incorporated into the system dynamics. Two opponent policies are implemented:

1. **Truth-Telling Policy** (`TruthTellingPolicy`):
   - Makes the lowest truthful legal bid when possible
   - Falls back to lowest legal bid (even if lying) when no truthful bid exists
   - Calls LIAR when the last bid is impossible given their hand
   - This policy is deterministic given the opponent's hand and bid history

2. **Optimal Policy** (optional):
   - Uses the same DP-computed optimal policy as Player 1
   - Provides a stronger adversary for training

The adversary policy $\pi_2$ is incorporated into the transition dynamics:

- When Player 1 takes action $a_1$, the system transitions deterministically
- Then Player 2's action $a_2 = \pi_2(s_2)$ is computed, where $s_2$ is the state from Player 2's perspective
- The combined transition $(s, a_1) \rightarrow (s', a_2) \rightarrow s''$ represents one full turn

### End-to-End Simulator

The simulator (`GameSimulation` class) implements a complete modular system:

**Parameters:**

- `num_dice_per_player` $N$: Number of dice per player (problem size)
- `player2_policy_type`: Adversary policy type ("truth_telling" or "optimal")
- Starting player: Randomly chosen or specified

**Features:**

- Modular action interface: accepts arbitrary actions (bids or LIAR calls)
- Returns both ground truth state ($H_1, H_2, D, G_o$) and observable state ($H_1, D, G_o$)
- Deterministic dynamics: dice hands remain constant throughout a game
- Noise/uncertainty: opponent's hand is hidden (observation uncertainty)
- Game termination: when LIAR is called, winner determined by `check_win_condition`

**Simulation Flow:**

1. Initialize: Roll dice for both players, create initial state
2. Loop while $G_o = \text{Active}$:
   - Current player observes their partial state
   - Player selects action (bid or call LIAR)
   - If bid: append to $D$, switch players
   - If LIAR: set $G_o = \text{GameOver}$, compute winner
3. Return: $(winner, final\_state)$

The simulator supports both deterministic evaluation (using ground truth state) and noisy evaluation (using observations only).

### Q-Learning with Function Approximation

**Function Approximation:**

- **Basis functions**: Feature vector $\phi(s, a)$ encoding:
  - Hand features: counts of each die value in $H_1$
  - Bid history features: last bid value, last bid amount, bid history length
  - Action features: bid value, bid amount (or indicator for LIAR call)
  - Derived features: $P_{valid}$ (probability last bid is valid), expected opponent dice counts
- **Q-value approximation**: $Q(s, a; \theta) = \theta^T \phi(s, a)$
  - Linear function approximation with parameter vector $\theta$
  - Alternative: Neural network $Q(s, a; \theta)$ with hidden layers

**Q-Learning Algorithm:**

- Experience replay buffer: stores transitions $(s, a, r, s', o')$
- Target Q-value: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
- Loss: $L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]$
- Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

**Hyperparameters:**

- Learning rate $\alpha$ (gradient step size)
- Discount factor $\gamma$ (from MATH.md)
- Exploration rate $\epsilon$: $\epsilon$-greedy policy ($\pi(a|s) = \epsilon/|A| + (1-\epsilon)\mathbf{1}[a = \arg\max_a Q(s,a)]$)
- Replay buffer size
- Batch size for experience replay
- Target network update frequency

**Training:**

- Run multiple games (episodes) collecting trajectories
- Store transitions in replay buffer
- Sample batches, update Q-function
- Track win rate over training iterations
- Demonstrate performance improvement with increased training iterations

### Policy Gradient Approach

**Stochastic Policy:**

- Policy $\pi(a|s; \theta)$: probability distribution over actions given state
- Parameterized by $\theta$ (e.g., neural network weights)
- Output: softmax over legal actions

**Function Approximation:**

- Policy network: $\pi(a|s; \theta) = \text{softmax}(f_\theta(s, a))$
- Features: same as Q-learning (hand, bid history, action features)
- Ensures $\sum_{a \in \text{LEGAL}(s)} \pi(a|s; \theta) = 1$

**Policy Gradient Algorithm:**

- Trajectory: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$
- Return: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$
- Baseline: $\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch)
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|s_t; \theta)]$
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Implementation:**

- Collect trajectories by running policy $\pi(a|s; \theta)$
- Compute returns $R(\tau)$ for each trajectory
- Subtract baseline (mean return) to reduce variance
- Update policy parameters using gradient ascent
- Validate policy: test in isolation and connected to simulator

**Hyperparameters:**

- Learning rate $\alpha$
- Discount factor $\gamma$
- Batch size (number of trajectories per update)
- Baseline type (mean return vs. value function baseline)

**Training:**

- Run policy gradient updates over multiple training runs
- Track win rate improvement over iterations
- Demonstrate convergence and performance gains

### Comparison of Approaches

**Computational Complexity:**

- **Q-Learning**:
  - Per update: $O(|\text{batch}| \cdot |\text{features}|)$ for linear, $O(|\text{batch}| \cdot |\text{network}|)$ for neural
  - Requires computing $\max_{a'} Q(s', a')$ over all legal actions
- **Policy Gradient**:
  - Per update: $O(|\text{trajectories}| \cdot |\text{avg\_length}| \cdot |\text{features}|)$
  - Requires full trajectory rollouts (no bootstrapping)

**Runtime:**

- Q-Learning: Can use off-policy data, more sample-efficient
- Policy Gradient: Requires on-policy data, may need more samples

**Performance:**

- Q-Learning: Learns value function, can be more stable
- Policy Gradient: Directly optimizes policy, may converge to better policies

**Convergence Rate:**

- Q-Learning: Depends on exploration strategy, may converge faster with good exploration
- Policy Gradient: May require more iterations but can find better local optima

**Trade-offs:**

- Q-Learning: Better for discrete action spaces, off-policy learning
- Policy Gradient: Better for stochastic policies, continuous action spaces, direct policy optimization
```

Updated MATH2.md with a reinforcement learning section. It includes:

1. **Mathematical formulation** — References MATH.md for core definitions (state space, action space, dynamics, sensor model, reward) without duplication.

2. **Adversary policy** — Describes the heuristic opponent policies (Truth-Telling and Optimal) and how they're incorporated into system dynamics.

3. **End-to-end simulator** — Documents the modular simulator with parameters, features, and simulation flow.

4. **Q-learning with function approximation** — Covers:
   - Feature basis functions
   - Q-value approximation
   - Algorithm with experience replay
   - Hyperparameters
   - Training procedure

5. **Policy gradient approach** — Covers:
   - Stochastic policy parameterization
   - Policy gradient algorithm with baseline
   - Implementation details
   - Hyperparameters
   - Training procedure

6. **Comparison** — Compares computational complexity, runtime, performance, and convergence rate between the two approaches.

The document references MATH.md for existing formulations and focuses on RL-specific extensions and implementations.

---

**User**

this MATH2.md should be more focused on math more than actual text or pseudocode etc.

---

**Cursor**

Rewriting MATH2.md to focus on mathematical formulations, using formal notation and equations instead of prose.

Rewriting MATH2.md to be more mathematical, using formal notation and equations:

```markdown
## Liar's Dice w/ Reinforcement Learning

Core mathematical formulation (state space $S$, action space $A$, system dynamics $T(s' \mid s, a)$, sensor model $\Omega(o \mid s', a)$, reward $R(s, a)$) defined in MATH.md. Extensions for RL:

### Extended System Dynamics with Adversary

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$ is Player 2's observable state.

**Truth-Telling Policy:**
$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{IMPOSSIBLE}(D_{\text{last}}, H_2) \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} \text{TRUTHFUL}(a, H_2) & \text{if exists} \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} a & \text{otherwise}
\end{cases}$

where:

- $\text{IMPOSSIBLE}(D_{\text{last}}, H_2) = \text{True}$ if $\text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount}$
- $\text{TRUTHFUL}(a, H_2) = \text{True}$ if $\text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}$

**Extended Transition:**
$T_{\text{extended}}(s'' \mid s, a_1) = T(s' \mid s, a_1) \cdot \mathbf{1}[a_2 = \pi_2(o_2')] \cdot T(s'' \mid s', a_2)$

where $o_2' = (H_2, D', G_o')$ is Player 2's observation after $a_1$.

### Adversary Policy

The opponent (Player 2) uses a heuristic policy that is incorporated into the system dynamics. Two opponent policies are implemented:

1. **Truth-Telling Policy** (`TruthTellingPolicy`):
   - Makes the lowest truthful legal bid when possible
   - Falls back to lowest legal bid (even if lying) when no truthful bid exists
   - Calls LIAR when the last bid is impossible given their hand
   - This policy is deterministic given the opponent's hand and bid history

2. **Optimal Policy** (optional):
   - Uses the same DP-computed optimal policy as Player 1
   - Provides a stronger adversary for training

The adversary policy $\pi_2$ is incorporated into the transition dynamics:

- When Player 1 takes action $a_1$, the system transitions deterministically
- Then Player 2's action $a_2 = \pi_2(s_2)$ is computed, where $s_2$ is the state from Player 2's perspective
- The combined transition $(s, a_1) \rightarrow (s', a_2) \rightarrow s''$ represents one full turn

### End-to-End Simulator

The simulator (`GameSimulation` class) implements a complete modular system:

**Parameters:**

- `num_dice_per_player` $N$: Number of dice per player (problem size)
- `player2_policy_type`: Adversary policy type ("truth_telling" or "optimal")
- Starting player: Randomly chosen or specified

**Features:**

- Modular action interface: accepts arbitrary actions (bids or LIAR calls)
- Returns both ground truth state ($H_1, H_2, D, G_o$) and observable state ($H_1, D, G_o$)
- Deterministic dynamics: dice hands remain constant throughout a game
- Noise/uncertainty: opponent's hand is hidden (observation uncertainty)
- Game termination: when LIAR is called, winner determined by `check_win_condition`

**Simulation Flow:**

1. Initialize: Roll dice for both players, create initial state
2. Loop while $G_o = \text{Active}$:
   - Current player observes their partial state
   - Player selects action (bid or call LIAR)
   - If bid: append to $D$, switch players
   - If LIAR: set $G_o = \text{GameOver}$, compute winner
3. Return: $(winner, final\_state)$

The simulator supports both deterministic evaluation (using ground truth state) and noisy evaluation (using observations only).

### Q-Learning with Function Approximation

**Function Approximation:**

- **Basis functions**: Feature vector $\phi(s, a)$ encoding:
  - Hand features: counts of each die value in $H_1$
  - Bid history features: last bid value, last bid amount, bid history length
  - Action features: bid value, bid amount (or indicator for LIAR call)
  - Derived features: $P_{valid}$ (probability last bid is valid), expected opponent dice counts
- **Q-value approximation**: $Q(s, a; \theta) = \theta^T \phi(s, a)$
  - Linear function approximation with parameter vector $\theta$
  - Alternative: Neural network $Q(s, a; \theta)$ with hidden layers

**Q-Learning Algorithm:**

- Experience replay buffer: stores transitions $(s, a, r, s', o')$
- Target Q-value: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
- Loss: $L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]$
- Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

**Hyperparameters:**

- Learning rate $\alpha$ (gradient step size)
- Discount factor $\gamma$ (from MATH.md)
- Exploration rate $\epsilon$: $\epsilon$-greedy policy ($\pi(a|s) = \epsilon/|A| + (1-\epsilon)\mathbf{1}[a = \arg\max_a Q(s,a)]$)
- Replay buffer size
- Batch size for experience replay
- Target network update frequency

**Training:**

- Run multiple games (episodes) collecting trajectories
- Store transitions in replay buffer
- Sample batches, update Q-function
- Track win rate over training iterations
- Demonstrate performance improvement with increased training iterations

### Policy Gradient Approach

**Stochastic Policy:**

- Policy $\pi(a|s; \theta)$: probability distribution over actions given state
- Parameterized by $\theta$ (e.g., neural network weights)
- Output: softmax over legal actions

**Function Approximation:**

- Policy network: $\pi(a|s; \theta) = \text{softmax}(f_\theta(s, a))$
- Features: same as Q-learning (hand, bid history, action features)
- Ensures $\sum_{a \in \text{LEGAL}(s)} \pi(a|s; \theta) = 1$

**Policy Gradient Algorithm:**

- Trajectory: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$
- Return: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$
- Baseline: $\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch)
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|s_t; \theta)]$
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Implementation:**

- Collect trajectories by running policy $\pi(a|s; \theta)$
- Compute returns $R(\tau)$ for each trajectory
- Subtract baseline (mean return) to reduce variance
- Update policy parameters using gradient ascent
- Validate policy: test in isolation and connected to simulator

**Hyperparameters:**

- Learning rate $\alpha$
- Discount factor $\gamma$
- Batch size (number of trajectories per update)
- Baseline type (mean return vs. value function baseline)

**Training:**

- Run policy gradient updates over multiple training runs
- Track win rate improvement over iterations
- Demonstrate convergence and performance gains

### Comparison of Approaches

**Computational Complexity:**

- **Q-Learning**:
  - Per update: $O(|\text{batch}| \cdot |\text{features}|)$ for linear, $O(|\text{batch}| \cdot |\text{network}|)$ for neural
  - Requires computing $\max_{a'} Q(s', a')$ over all legal actions
- **Policy Gradient**:
  - Per update: $O(|\text{trajectories}| \cdot |\text{avg\_length}| \cdot |\text{features}|)$
  - Requires full trajectory rollouts (no bootstrapping)

**Runtime:**

- Q-Learning: Can use off-policy data, more sample-efficient
- Policy Gradient: Requires on-policy data, may need more samples

**Performance:**

- Q-Learning: Learns value function, can be more stable
- Policy Gradient: Directly optimizes policy, may converge to better policies

**Convergence Rate:**

- Q-Learning: Depends on exploration strategy, may converge faster with good exploration
- Policy Gradient: May require more iterations but can find better local optima

**Trade-offs:**

- Q-Learning: Better for discrete action spaces, off-policy learning
- Policy Gradient: Better for stochastic policies, continuous action spaces, direct policy optimization
```

```markdown
## Liar's Dice w/ Reinforcement Learning

Core mathematical formulation (state space $S$, action space $A$, system dynamics $T(s' \mid s, a)$, sensor model $\Omega(o \mid s', a)$, reward $R(s, a)$) defined in MATH.md. Extensions for RL:

### Extended System Dynamics with Adversary

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$ is Player 2's observable state.

**Truth-Telling Policy:**
$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{IMPOSSIBLE}(D_{\text{last}}, H_2) \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} \text{TRUTHFUL}(a, H_2) & \text{if exists} \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} a & \text{otherwise}
\end{cases}$

where:

- $\text{IMPOSSIBLE}(D_{\text{last}}, H_2) = \text{True}$ if $\text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount}$
- $\text{TRUTHFUL}(a, H_2) = \text{True}$ if $\text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}$

**Extended Transition:**
$T_{\text{extended}}(s'' \mid s, a_1) = T(s' \mid s, a_1) \cdot \mathbf{1}[a_2 = \pi_2(o_2')] \cdot T(s'' \mid s', a_2)$

where $o_2' = (H_2, D', G_o')$ is Player 2's observation after $a_1$.

### Multi-Episode Extension

For RL training over $M$ episodes:

**Episode initialization:**
$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

**Episode trajectory:**
$\tau^{(m)} = (s_0^{(m)}, a_0^{(m)}, r_0^{(m)}, o_1^{(m)}, s_1^{(m)}, a_1^{(m)}, r_1^{(m)}, ..., s_T^{(m)})$

where $s_t^{(m)} = (H_1^{(m)}, H_2^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$ and $o_t^{(m)} = (H_1^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$.

**Simulator outputs:**

- Ground truth: $s_t^{(m)}$ (full state)
- Observation: $o_t^{(m)}$ (partial state)
- Reward: $r_t^{(m)} = R(s_t^{(m)}, a_t^{(m)})$ (from MATH.md)

### Q-Learning with Function Approximation

**Function Approximation:**

- **Basis functions**: Feature vector $\phi(s, a)$ encoding:
  - Hand features: counts of each die value in $H_1$
  - Bid history features: last bid value, last bid amount, bid history length
  - Action features: bid value, bid amount (or indicator for LIAR call)
  - Derived features: $P_{valid}$ (probability last bid is valid), expected opponent dice counts
- **Q-value approximation**: $Q(s, a; \theta) = \theta^T \phi(s, a)$
  - Linear function approximation with parameter vector $\theta$
  - Alternative: Neural network $Q(s, a; \theta)$ with hidden layers

**Q-Learning Algorithm:**

- Experience replay buffer: stores transitions $(s, a, r, s', o')$
- Target Q-value: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
- Loss: $L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]$
- Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

**Hyperparameters:**

- Learning rate $\alpha$ (gradient step size)
- Discount factor $\gamma$ (from MATH.md)
- Exploration rate $\epsilon$: $\epsilon$-greedy policy ($\pi(a|s) = \epsilon/|A| + (1-\epsilon)\mathbf{1}[a = \arg\max_a Q(s,a)]$)
- Replay buffer size
- Batch size for experience replay
- Target network update frequency

**Training:**

- Run multiple games (episodes) collecting trajectories
- Store transitions in replay buffer
- Sample batches, update Q-function
- Track win rate over training iterations
- Demonstrate performance improvement with increased training iterations

### Policy Gradient Approach

**Stochastic Policy:**

- Policy $\pi(a|s; \theta)$: probability distribution over actions given state
- Parameterized by $\theta$ (e.g., neural network weights)
- Output: softmax over legal actions

**Function Approximation:**

- Policy network: $\pi(a|s; \theta) = \text{softmax}(f_\theta(s, a))$
- Features: same as Q-learning (hand, bid history, action features)
- Ensures $\sum_{a \in \text{LEGAL}(s)} \pi(a|s; \theta) = 1$

**Policy Gradient Algorithm:**

- Trajectory: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$
- Return: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$
- Baseline: $\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch)
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|s_t; \theta)]$
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Implementation:**

- Collect trajectories by running policy $\pi(a|s; \theta)$
- Compute returns $R(\tau)$ for each trajectory
- Subtract baseline (mean return) to reduce variance
- Update policy parameters using gradient ascent
- Validate policy: test in isolation and connected to simulator

**Hyperparameters:**

- Learning rate $\alpha$
- Discount factor $\gamma$
- Batch size (number of trajectories per update)
- Baseline type (mean return vs. value function baseline)

**Training:**

- Run policy gradient updates over multiple training runs
- Track win rate improvement over iterations
- Demonstrate convergence and performance gains

### Comparison of Approaches

**Computational Complexity:**

- **Q-Learning**:
  - Per update: $O(|\text{batch}| \cdot |\text{features}|)$ for linear, $O(|\text{batch}| \cdot |\text{network}|)$ for neural
  - Requires computing $\max_{a'} Q(s', a')$ over all legal actions
- **Policy Gradient**:
  - Per update: $O(|\text{trajectories}| \cdot |\text{avg\_length}| \cdot |\text{features}|)$
  - Requires full trajectory rollouts (no bootstrapping)

**Runtime:**

- Q-Learning: Can use off-policy data, more sample-efficient
- Policy Gradient: Requires on-policy data, may need more samples

**Performance:**

- Q-Learning: Learns value function, can be more stable
- Policy Gradient: Directly optimizes policy, may converge to better policies

**Convergence Rate:**

- Q-Learning: Depends on exploration strategy, may converge faster with good exploration
- Policy Gradient: May require more iterations but can find better local optima

**Trade-offs:**

- Q-Learning: Better for discrete action spaces, off-policy learning
- Policy Gradient: Better for stochastic policies, continuous action spaces, direct policy optimization
```

```markdown
## Liar's Dice w/ Reinforcement Learning

Core mathematical formulation (state space $S$, action space $A$, system dynamics $T(s' \mid s, a)$, sensor model $\Omega(o \mid s', a)$, reward $R(s, a)$) defined in MATH.md. Extensions for RL:

### Extended System Dynamics with Adversary

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$ is Player 2's observable state.

**Truth-Telling Policy:**
$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{IMPOSSIBLE}(D_{\text{last}}, H_2) \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} \text{TRUTHFUL}(a, H_2) & \text{if exists} \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} a & \text{otherwise}
\end{cases}$

where:

- $\text{IMPOSSIBLE}(D_{\text{last}}, H_2) = \text{True}$ if $\text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount}$
- $\text{TRUTHFUL}(a, H_2) = \text{True}$ if $\text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}$

**Extended Transition:**
$T_{\text{extended}}(s'' \mid s, a_1) = T(s' \mid s, a_1) \cdot \mathbf{1}[a_2 = \pi_2(o_2')] \cdot T(s'' \mid s', a_2)$

where $o_2' = (H_2, D', G_o')$ is Player 2's observation after $a_1$.

### Multi-Episode Extension

For RL training over $M$ episodes:

**Episode initialization:**
$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

**Episode trajectory:**
$\tau^{(m)} = (s_0^{(m)}, a_0^{(m)}, r_0^{(m)}, o_1^{(m)}, s_1^{(m)}, a_1^{(m)}, r_1^{(m)}, ..., s_T^{(m)})$

where $s_t^{(m)} = (H_1^{(m)}, H_2^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$ and $o_t^{(m)} = (H_1^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$.

**Simulator outputs:**

- Ground truth: $s_t^{(m)}$ (full state)
- Observation: $o_t^{(m)}$ (partial state)
- Reward: $r_t^{(m)} = R(s_t^{(m)}, a_t^{(m)})$ (from MATH.md)

### Q-Learning with Function Approximation

**Feature Vector:**
$\phi(o, a) \in \mathbb{R}^d$ where:

- $\phi_1(o, a), ..., \phi_6(o, a)$: counts of dice values 1-6 in $H_1$
- $\phi_7(o, a)$: $D_{\text{last}}.\text{Value}$ if $D \ne \emptyset$, else 0
- $\phi_8(o, a)$: $D_{\text{last}}.\text{Amount}$ if $D \ne \emptyset$, else 0
- $\phi_9(o, a)$: $|D|$ (bid history length)
- $\phi_{10}(o, a)$: $P_{valid}(D_{\text{last}} \mid H_1)$ (from MATH.md line 102)
- $\phi_{11}(o, a)$: $a.\text{Value}$ if $a \in B$, else 0
- $\phi_{12}(o, a)$: $a.\text{Amount}$ if $a \in B$, else 0
- $\phi_{13}(o, a)$: $\mathbf{1}[a = C]$ (LIAR indicator)

**Q-Function Approximation:**
$Q(o, a; \theta) = \theta^T \phi(o, a)$ (linear) or $Q(o, a; \theta) = f_\theta(\phi(o, a))$ (neural network)

**Experience Replay Buffer:**
$\mathcal{D} = \{(o_i, a_i, r_i, o_i', s_i')\}_{i=1}^{|\mathcal{D}|}$ where transitions sampled from trajectories $\tau^{(m)}$.

**Q-Learning Update:**
Target: $y_i = r_i + \gamma \max_{a' \in \text{LEGAL}(s_i')} Q(o_i', a'; \theta^-)$

Loss: $L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(o_i, a_i, r_i, o_i', s_i') \in \mathcal{B}} (y_i - Q(o_i, a_i; \theta))^2$

Gradient: $\nabla_\theta L(\theta) = \frac{1}{|\mathcal{B}|} \sum_i (Q(o_i, a_i; \theta) - y_i) \nabla_\theta Q(o_i, a_i; \theta)$

Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

where $\mathcal{B} \subset \mathcal{D}$ is a batch, $\theta^-$ is target network parameters.

**Exploration Policy:**
$\pi_{\epsilon}(a|o; \theta) = \begin{cases}
\text{Uniform}(\text{LEGAL}(o)) & \text{with prob } \epsilon \\
\mathbf{1}[a = \arg\max_{a' \in \text{LEGAL}(o)} Q(o, a'; \theta)] & \text{with prob } 1-\epsilon
\end{cases}$

**Hyperparameters:**

- $\alpha \in \mathbb{R}^+$: learning rate
- $\gamma \in [0, 1]$: discount factor (from MATH.md)
- $\epsilon \in [0, 1]$: exploration rate
- $|\mathcal{D}|$: replay buffer size
- $|\mathcal{B}|$: batch size
- $\tau_{\text{update}}$: target network update frequency

### Policy Gradient Approach

**Stochastic Policy:**

- Policy $\pi(a|s; \theta)$: probability distribution over actions given state
- Parameterized by $\theta$ (e.g., neural network weights)
- Output: softmax over legal actions

**Function Approximation:**

- Policy network: $\pi(a|s; \theta) = \text{softmax}(f_\theta(s, a))$
- Features: same as Q-learning (hand, bid history, action features)
- Ensures $\sum_{a \in \text{LEGAL}(s)} \pi(a|s; \theta) = 1$

**Policy Gradient Algorithm:**

- Trajectory: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)$
- Return: $R(\tau) = \sum_{t=0}^T \gamma^t r_t$
- Baseline: $\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch)
- Policy gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|s_t; \theta)]$
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Implementation:**

- Collect trajectories by running policy $\pi(a|s; \theta)$
- Compute returns $R(\tau)$ for each trajectory
- Subtract baseline (mean return) to reduce variance
- Update policy parameters using gradient ascent
- Validate policy: test in isolation and connected to simulator

**Hyperparameters:**

- Learning rate $\alpha$
- Discount factor $\gamma$
- Batch size (number of trajectories per update)
- Baseline type (mean return vs. value function baseline)

**Training:**

- Run policy gradient updates over multiple training runs
- Track win rate improvement over iterations
- Demonstrate convergence and performance gains

### Comparison of Approaches

**Computational Complexity:**

- **Q-Learning**:
  - Per update: $O(|\text{batch}| \cdot |\text{features}|)$ for linear, $O(|\text{batch}| \cdot |\text{network}|)$ for neural
  - Requires computing $\max_{a'} Q(s', a')$ over all legal actions
- **Policy Gradient**:
  - Per update: $O(|\text{trajectories}| \cdot |\text{avg\_length}| \cdot |\text{features}|)$
  - Requires full trajectory rollouts (no bootstrapping)

**Runtime:**

- Q-Learning: Can use off-policy data, more sample-efficient
- Policy Gradient: Requires on-policy data, may need more samples

**Performance:**

- Q-Learning: Learns value function, can be more stable
- Policy Gradient: Directly optimizes policy, may converge to better policies

**Convergence Rate:**

- Q-Learning: Depends on exploration strategy, may converge faster with good exploration
- Policy Gradient: May require more iterations but can find better local optima

**Trade-offs:**

- Q-Learning: Better for discrete action spaces, off-policy learning
- Policy Gradient: Better for stochastic policies, continuous action spaces, direct policy optimization
```

```markdown
## Liar's Dice w/ Reinforcement Learning

Core mathematical formulation (state space $S$, action space $A$, system dynamics $T(s' \mid s, a)$, sensor model $\Omega(o \mid s', a)$, reward $R(s, a)$) defined in MATH.md. Extensions for RL:

### Extended System Dynamics with Adversary

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$ is Player 2's observable state.

**Truth-Telling Policy:**
$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{IMPOSSIBLE}(D_{\text{last}}, H_2) \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} \text{TRUTHFUL}(a, H_2) & \text{if exists} \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} a & \text{otherwise}
\end{cases}$

where:

- $\text{IMPOSSIBLE}(D_{\text{last}}, H_2) = \text{True}$ if $\text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount}$
- $\text{TRUTHFUL}(a, H_2) = \text{True}$ if $\text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}$

**Extended Transition:**
$T_{\text{extended}}(s'' \mid s, a_1) = T(s' \mid s, a_1) \cdot \mathbf{1}[a_2 = \pi_2(o_2')] \cdot T(s'' \mid s', a_2)$

where $o_2' = (H_2, D', G_o')$ is Player 2's observation after $a_1$.

### Multi-Episode Extension

For RL training over $M$ episodes:

**Episode initialization:**
$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

**Episode trajectory:**
$\tau^{(m)} = (s_0^{(m)}, a_0^{(m)}, r_0^{(m)}, o_1^{(m)}, s_1^{(m)}, a_1^{(m)}, r_1^{(m)}, ..., s_T^{(m)})$

where $s_t^{(m)} = (H_1^{(m)}, H_2^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$ and $o_t^{(m)} = (H_1^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$.

**Simulator outputs:**

- Ground truth: $s_t^{(m)}$ (full state)
- Observation: $o_t^{(m)}$ (partial state)
- Reward: $r_t^{(m)} = R(s_t^{(m)}, a_t^{(m)})$ (from MATH.md)

### Q-Learning with Function Approximation

**Feature Vector:**
$\phi(o, a) \in \mathbb{R}^d$ where:

- $\phi_1(o, a), ..., \phi_6(o, a)$: counts of dice values 1-6 in $H_1$
- $\phi_7(o, a)$: $D_{\text{last}}.\text{Value}$ if $D \ne \emptyset$, else 0
- $\phi_8(o, a)$: $D_{\text{last}}.\text{Amount}$ if $D \ne \emptyset$, else 0
- $\phi_9(o, a)$: $|D|$ (bid history length)
- $\phi_{10}(o, a)$: $P_{valid}(D_{\text{last}} \mid H_1)$ (from MATH.md line 102)
- $\phi_{11}(o, a)$: $a.\text{Value}$ if $a \in B$, else 0
- $\phi_{12}(o, a)$: $a.\text{Amount}$ if $a \in B$, else 0
- $\phi_{13}(o, a)$: $\mathbf{1}[a = C]$ (LIAR indicator)

**Q-Function Approximation:**
$Q(o, a; \theta) = \theta^T \phi(o, a)$ (linear) or $Q(o, a; \theta) = f_\theta(\phi(o, a))$ (neural network)

**Experience Replay Buffer:**
$\mathcal{D} = \{(o_i, a_i, r_i, o_i', s_i')\}_{i=1}^{|\mathcal{D}|}$ where transitions sampled from trajectories $\tau^{(m)}$.

**Q-Learning Update:**
Target: $y_i = r_i + \gamma \max_{a' \in \text{LEGAL}(s_i')} Q(o_i', a'; \theta^-)$

Loss: $L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(o_i, a_i, r_i, o_i', s_i') \in \mathcal{B}} (y_i - Q(o_i, a_i; \theta))^2$

Gradient: $\nabla_\theta L(\theta) = \frac{1}{|\mathcal{B}|} \sum_i (Q(o_i, a_i; \theta) - y_i) \nabla_\theta Q(o_i, a_i; \theta)$

Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

where $\mathcal{B} \subset \mathcal{D}$ is a batch, $\theta^-$ is target network parameters.

**Exploration Policy:**
$\pi_{\epsilon}(a|o; \theta) = \begin{cases}
\text{Uniform}(\text{LEGAL}(o)) & \text{with prob } \epsilon \\
\mathbf{1}[a = \arg\max_{a' \in \text{LEGAL}(o)} Q(o, a'; \theta)] & \text{with prob } 1-\epsilon
\end{cases}$

**Hyperparameters:**

- $\alpha \in \mathbb{R}^+$: learning rate
- $\gamma \in [0, 1]$: discount factor (from MATH.md)
- $\epsilon \in [0, 1]$: exploration rate
- $|\mathcal{D}|$: replay buffer size
- $|\mathcal{B}|$: batch size
- $\tau_{\text{update}}$: target network update frequency

### Policy Gradient Approach

**Stochastic Policy:**
$\pi(a|o; \theta) = \frac{\exp(f_\theta(\phi(o, a)))}{\sum_{a' \in \text{LEGAL}(o)} \exp(f_\theta(\phi(o, a')))}$

where $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}$ is parameterized function (e.g., neural network), ensuring $\sum_{a \in \text{LEGAL}(o)} \pi(a|o; \theta) = 1$.

**Trajectory Return:**
$R(\tau) = \sum_{t=0}^T \gamma^t r_t$ where $\tau = (s_0, a_0, r_0, o_1, s_1, a_1, r_1, ..., s_T)$.

**Baseline:**
$\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch of $N$ trajectories).

**Policy Gradient:**
Objective: $J(\theta) = \mathbb{E}_{\tau \sim \pi(\cdot|\cdot; \theta)}[R(\tau)]$

Gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|o_t; \theta)\right]$

Monte Carlo estimate: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} (R(\tau_i) - \bar{R}) \nabla_\theta \log \pi(a_{i,t}|o_{i,t}; \theta)$

Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Hyperparameters:**

- $\alpha \in \mathbb{R}^+$: learning rate
- $\gamma \in [0, 1]$: discount factor
- $N$: batch size (number of trajectories per update)
- Baseline: $\bar{R}$ (mean return) or $V(o_t; \phi)$ (value function baseline)

### Comparison of Approaches

**Computational Complexity:**

- **Q-Learning**:
  - Per update: $O(|\text{batch}| \cdot |\text{features}|)$ for linear, $O(|\text{batch}| \cdot |\text{network}|)$ for neural
  - Requires computing $\max_{a'} Q(s', a')$ over all legal actions
- **Policy Gradient**:
  - Per update: $O(|\text{trajectories}| \cdot |\text{avg\_length}| \cdot |\text{features}|)$
  - Requires full trajectory rollouts (no bootstrapping)

**Runtime:**

- Q-Learning: Can use off-policy data, more sample-efficient
- Policy Gradient: Requires on-policy data, may need more samples

**Performance:**

- Q-Learning: Learns value function, can be more stable
- Policy Gradient: Directly optimizes policy, may converge to better policies

**Convergence Rate:**

- Q-Learning: Depends on exploration strategy, may converge faster with good exploration
- Policy Gradient: May require more iterations but can find better local optima

**Trade-offs:**

- Q-Learning: Better for discrete action spaces, off-policy learning
- Policy Gradient: Better for stochastic policies, continuous action spaces, direct policy optimization
```

```markdown
## Liar's Dice w/ Reinforcement Learning

Core mathematical formulation (state space $S$, action space $A$, system dynamics $T(s' \mid s, a)$, sensor model $\Omega(o \mid s', a)$, reward $R(s, a)$) defined in MATH.md. Extensions for RL:

### Extended System Dynamics with Adversary

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$ is Player 2's observable state.

**Truth-Telling Policy:**
$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{IMPOSSIBLE}(D_{\text{last}}, H_2) \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} \text{TRUTHFUL}(a, H_2) & \text{if exists} \\
\arg\min_{a \in \text{LEGAL}(D_{\text{last}})} a & \text{otherwise}
\end{cases}$

where:

- $\text{IMPOSSIBLE}(D_{\text{last}}, H_2) = \text{True}$ if $\text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount}$
- $\text{TRUTHFUL}(a, H_2) = \text{True}$ if $\text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}$

**Extended Transition:**
$T_{\text{extended}}(s'' \mid s, a_1) = T(s' \mid s, a_1) \cdot \mathbf{1}[a_2 = \pi_2(o_2')] \cdot T(s'' \mid s', a_2)$

where $o_2' = (H_2, D', G_o')$ is Player 2's observation after $a_1$.

### Multi-Episode Extension

For RL training over $M$ episodes:

**Episode initialization:**
$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

**Episode trajectory:**
$\tau^{(m)} = (s_0^{(m)}, a_0^{(m)}, r_0^{(m)}, o_1^{(m)}, s_1^{(m)}, a_1^{(m)}, r_1^{(m)}, ..., s_T^{(m)})$

where $s_t^{(m)} = (H_1^{(m)}, H_2^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$ and $o_t^{(m)} = (H_1^{(m)}, D_t^{(m)}, G_{o,t}^{(m)})$.

**Simulator outputs:**

- Ground truth: $s_t^{(m)}$ (full state)
- Observation: $o_t^{(m)}$ (partial state)
- Reward: $r_t^{(m)} = R(s_t^{(m)}, a_t^{(m)})$ (from MATH.md)

### Q-Learning with Function Approximation

**Feature Vector:**
$\phi(o, a) \in \mathbb{R}^d$ where:

- $\phi_1(o, a), ..., \phi_6(o, a)$: counts of dice values 1-6 in $H_1$
- $\phi_7(o, a)$: $D_{\text{last}}.\text{Value}$ if $D \ne \emptyset$, else 0
- $\phi_8(o, a)$: $D_{\text{last}}.\text{Amount}$ if $D \ne \emptyset$, else 0
- $\phi_9(o, a)$: $|D|$ (bid history length)
- $\phi_{10}(o, a)$: $P_{valid}(D_{\text{last}} \mid H_1)$ (from MATH.md line 102)
- $\phi_{11}(o, a)$: $a.\text{Value}$ if $a \in B$, else 0
- $\phi_{12}(o, a)$: $a.\text{Amount}$ if $a \in B$, else 0
- $\phi_{13}(o, a)$: $\mathbf{1}[a = C]$ (LIAR indicator)

**Q-Function Approximation:**
$Q(o, a; \theta) = \theta^T \phi(o, a)$ (linear) or $Q(o, a; \theta) = f_\theta(\phi(o, a))$ (neural network)

**Experience Replay Buffer:**
$\mathcal{D} = \{(o_i, a_i, r_i, o_i', s_i')\}_{i=1}^{|\mathcal{D}|}$ where transitions sampled from trajectories $\tau^{(m)}$.

**Q-Learning Update:**
Target: $y_i = r_i + \gamma \max_{a' \in \text{LEGAL}(s_i')} Q(o_i', a'; \theta^-)$

Loss: $L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(o_i, a_i, r_i, o_i', s_i') \in \mathcal{B}} (y_i - Q(o_i, a_i; \theta))^2$

Gradient: $\nabla_\theta L(\theta) = \frac{1}{|\mathcal{B}|} \sum_i (Q(o_i, a_i; \theta) - y_i) \nabla_\theta Q(o_i, a_i; \theta)$

Update: $\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

where $\mathcal{B} \subset \mathcal{D}$ is a batch, $\theta^-$ is target network parameters.

**Exploration Policy:**
$\pi_{\epsilon}(a|o; \theta) = \begin{cases}
\text{Uniform}(\text{LEGAL}(o)) & \text{with prob } \epsilon \\
\mathbf{1}[a = \arg\max_{a' \in \text{LEGAL}(o)} Q(o, a'; \theta)] & \text{with prob } 1-\epsilon
\end{cases}$

**Hyperparameters:**

- $\alpha \in \mathbb{R}^+$: learning rate
- $\gamma \in [0, 1]$: discount factor (from MATH.md)
- $\epsilon \in [0, 1]$: exploration rate
- $|\mathcal{D}|$: replay buffer size
- $|\mathcal{B}|$: batch size
- $\tau_{\text{update}}$: target network update frequency

### Policy Gradient Approach

**Stochastic Policy:**
$\pi(a|o; \theta) = \frac{\exp(f_\theta(\phi(o, a)))}{\sum_{a' \in \text{LEGAL}(o)} \exp(f_\theta(\phi(o, a')))}$

where $f_\theta: \mathbb{R}^d \rightarrow \mathbb{R}$ is parameterized function (e.g., neural network), ensuring $\sum_{a \in \text{LEGAL}(o)} \pi(a|o; \theta) = 1$.

**Trajectory Return:**
$R(\tau) = \sum_{t=0}^T \gamma^t r_t$ where $\tau = (s_0, a_0, r_0, o_1, s_1, a_1, r_1, ..., s_T)$.

**Baseline:**
$\bar{R} = \frac{1}{N} \sum_{i=1}^N R(\tau_i)$ (mean return over batch of $N$ trajectories).

**Policy Gradient:**
Objective: $J(\theta) = \mathbb{E}_{\tau \sim \pi(\cdot|\cdot; \theta)}[R(\tau)]$

Gradient: $\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_{t=0}^T (R(\tau) - \bar{R}) \nabla_\theta \log \pi(a_t|o_t; \theta)\right]$

Monte Carlo estimate: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} (R(\tau_i) - \bar{R}) \nabla_\theta \log \pi(a_{i,t}|o_{i,t}; \theta)$

Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**Hyperparameters:**

- $\alpha \in \mathbb{R}^+$: learning rate
- $\gamma \in [0, 1]$: discount factor
- $N$: batch size (number of trajectories per update)
- Baseline: $\bar{R}$ (mean return) or $V(o_t; \phi)$ (value function baseline)

### Comparison of Approaches

**Computational Complexity:**

Q-Learning per update:

- Linear: $O(|\mathcal{B}| \cdot d)$ where $d = |\phi(o, a)|$
- Neural: $O(|\mathcal{B}| \cdot |\theta|)$ where $|\theta|$ is network size
- Action maximization: $O(|\text{LEGAL}(o')|)$ per sample

Policy Gradient per update:

- $O(N \cdot \bar{T} \cdot |\theta|)$ where $N$ is batch size, $\bar{T}$ is average trajectory length

**Sample Complexity:**

Q-Learning: Off-policy, sample efficiency $\tilde{O}(|\mathcal{D}|^{-1/2})$ with replay buffer.

Policy Gradient: On-policy, requires fresh trajectories per update, sample efficiency $\tilde{O}(N^{-1/2})$.

**Convergence:**

Q-Learning: Under conditions (sufficient exploration, function approximation capacity), converges to $Q^*$ with rate $O(\alpha^k)$ where $k$ is iteration count.

Policy Gradient: Converges to local optimum of $J(\theta)$ with rate $O(\alpha / \sqrt{N})$ under policy gradient theorem conditions.

**Performance Metrics:**

Win rate: $W(\theta) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[\text{Player 1 wins episode } m]$

Expected return: $\mathbb{E}[R(\tau)] = J(\theta)$

**Trade-offs:**

- Q-Learning: $|\text{LEGAL}(o)|$ discrete actions, off-policy, value-based
- Policy Gradient: Stochastic policy $\pi(a|o; \theta)$, on-policy, direct policy optimization
```

Rewrote MATH2.md to focus on mathematical formulations. Changes:

1. **Extended System Dynamics with Adversary** — Formal definition of adversary policy $\pi_2$ with mathematical conditions
2. **Multi-Episode Extension** — Mathematical notation for episode trajectories and simulator outputs
3. **Q-Learning** — Formal equations for:
   - Feature vector $\phi(o, a)$ with explicit components
   - Q-function approximation
   - Experience replay buffer notation
   - Q-learning update equations
   - Exploration policy definition
4. **Policy Gradient** — Mathematical definitions for:
   - Stochastic policy with softmax
   - Trajectory return
   - Policy gradient objective and gradient
   - Monte Carlo estimate
5. **Comparison** — Formal complexity analysis using Big-O notation and convergence rates

The document now uses formal mathematical notation and equations instead of prose descriptions, matching the style of MATH.md.

---

**User**

double check math2.md and ensure that its correct and focuses on math and is not adding random bullshit

---

**Cursor**

Issues I found:

1. **Extended Transition formula** — mathematically awkward, product of indicator/transitions doesn't make sense
2. **Feature Vector** — overly prescriptive (13 specific features is implementation detail, not math)
3. **Comparison section** — convergence rates and sample complexity claims are made up / not rigorous
4. **Truth-Telling Policy** — `argmin` over TRUTHFUL is wrong (TRUTHFUL is boolean, can't minimize over it)
5. **"Simulator outputs"** — implementation detail, not math

Let me clean it up:

```markdown
## Liar's Dice w/ Reinforcement Learning

Core formulation ($S$, $A$, $T(s' \mid s, a)$, $\Omega(o \mid s', a)$, $R(s, a)$) defined in MATH.md.

### Adversary Policy

Adversary policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$.

**Truth-Telling Policy:**

$\pi_2^{TT}(o_2) = \begin{cases}
C & \text{if } \text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount} \\
\min\{a \in \text{LEGAL}(D_{\text{last}}) : \text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}\} & \text{if exists} \\
\min\{a \in \text{LEGAL}(D_{\text{last}})\} & \text{otherwise}
\end{cases}$

The adversary policy is incorporated into the MDP: from Player 1's perspective, taking action $a_1$ leads to a new state after Player 2 responds with $a_2 = \pi_2(o_2)$.

### Multi-Episode Training

$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

Trajectory: $\tau^{(m)} = (o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T, a_T, r_T)$

where $o_t = (H_1, D_t, G_{o,t})$ and $r_t = R(s_t, a_t)$ (from MATH.md).

### Q-Learning with Function Approximation

**Function Approximation:**

$Q(o, a; \theta) = \theta^T \phi(o, a)$ where $\phi: O \times A \rightarrow \mathbb{R}^d$ is a feature basis.

**Q-Learning Update:**

$y_i = r_i + \gamma \max_{a' \in \text{LEGAL}(o_i')} Q(o_i', a'; \theta^-)$

$L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (y_i - Q(o_i, a_i; \theta))^2$

$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

where $\mathcal{B}$ is a batch sampled from replay buffer $\mathcal{D} = \{(o_i, a_i, r_i, o_i')\}$, and $\theta^-$ is target network.

**$\epsilon$-Greedy Exploration:**

$\pi_{\epsilon}(a|o; \theta) = \begin{cases}
\frac{\epsilon}{|\text{LEGAL}(o)|} + (1-\epsilon) & \text{if } a = \arg\max_{a'} Q(o, a'; \theta) \\
\frac{\epsilon}{|\text{LEGAL}(o)|} & \text{otherwise}
\end{cases}$

### Policy Gradient

**Stochastic Policy:**

$\pi(a|o; \theta) = \frac{\exp(f_\theta(o, a))}{\sum_{a' \in \text{LEGAL}(o)} \exp(f_\theta(o, a'))}$

**Trajectory Return:**

$G(\tau) = \sum_{t=0}^T \gamma^t r_t$

**REINFORCE with Baseline:**

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T (G(\tau) - b) \nabla_\theta \log \pi(a_t|o_t; \theta)\right]$

where $b = \frac{1}{N} \sum_{i=1}^N G(\tau_i)$ is the mean return baseline.

$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

### Hyperparameters

| Symbol            | Description                               |
| ----------------- | ----------------------------------------- |
| $\alpha$          | Learning rate                             |
| $\gamma$          | Discount factor                           |
| $\epsilon$        | Exploration rate (Q-learning)             |
| $\|\mathcal{B}\|$ | Batch size                                |
| $N$               | Trajectories per update (policy gradient) |

### Performance Metric

Win rate: $W(\theta) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[\text{P1 wins episode } m]$
```

Cleaned up. Removed:

1. **Fake convergence rates** — the $O(\alpha^k)$ and $\tilde{O}(|\mathcal{D}|^{-1/2})$ claims were not rigorous
2. **13-feature enumeration** — implementation detail, replaced with general $\phi: O \times A \rightarrow \mathbb{R}^d$
3. **Broken truth-telling formula** — fixed to use proper set notation with $\min\{a : \text{condition}\}$
4. **Extended transition formula** — was mathematically awkward, replaced with plain language
5. **"Simulator outputs"** section — implementation, not math
6. **Entire "Comparison of Approaches" section** — mostly made-up complexity claims

What remains is the core RL math: Q-learning update, $\epsilon$-greedy, policy gradient (REINFORCE), and the adversary policy definition.

---

**User**

in the reinforcement learning approach, we assume the opponent is always a truth telling opponent no matter what you don' thave to worry about opponent being optimal or whatever

---

**Cursor**

```markdown
## Liar's Dice w/ Reinforcement Learning

Core formulation ($S$, $A$, $T(s' \mid s, a)$, $\Omega(o \mid s', a)$, $R(s, a)$) defined in MATH.md.

### Adversary Policy (Fixed)

Opponent uses truth-telling policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$:

$\pi_2(o_2) = \begin{cases}
C & \text{if } \text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount} \\
\min\{a \in \text{LEGAL}(D_{\text{last}}) : \text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}\} & \text{if exists} \\
\min\{a \in \text{LEGAL}(D_{\text{last}})\} & \text{otherwise}
\end{cases}$

This fixed policy is incorporated into the MDP dynamics.

### Multi-Episode Training

$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

Trajectory: $\tau^{(m)} = (o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T, a_T, r_T)$

where $o_t = (H_1, D_t, G_{o,t})$ and $r_t = R(s_t, a_t)$ (from MATH.md).

### Q-Learning with Function Approximation

**Function Approximation:**

$Q(o, a; \theta) = \theta^T \phi(o, a)$ where $\phi: O \times A \rightarrow \mathbb{R}^d$ is a feature basis.

**Q-Learning Update:**

$y_i = r_i + \gamma \max_{a' \in \text{LEGAL}(o_i')} Q(o_i', a'; \theta^-)$

$L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (y_i - Q(o_i, a_i; \theta))^2$

$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$

where $\mathcal{B}$ is a batch sampled from replay buffer $\mathcal{D} = \{(o_i, a_i, r_i, o_i')\}$, and $\theta^-$ is target network.

**$\epsilon$-Greedy Exploration:**

$\pi_{\epsilon}(a|o; \theta) = \begin{cases}
\frac{\epsilon}{|\text{LEGAL}(o)|} + (1-\epsilon) & \text{if } a = \arg\max_{a'} Q(o, a'; \theta) \\
\frac{\epsilon}{|\text{LEGAL}(o)|} & \text{otherwise}
\end{cases}$

### Policy Gradient

**Stochastic Policy:**

$\pi(a|o; \theta) = \frac{\exp(f_\theta(o, a))}{\sum_{a' \in \text{LEGAL}(o)} \exp(f_\theta(o, a'))}$

**Trajectory Return:**

$G(\tau) = \sum_{t=0}^T \gamma^t r_t$

**REINFORCE with Baseline:**

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T (G(\tau) - b) \nabla_\theta \log \pi(a_t|o_t; \theta)\right]$

where $b = \frac{1}{N} \sum_{i=1}^N G(\tau_i)$ is the mean return baseline.

$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

### Hyperparameters

| Symbol            | Description                               |
| ----------------- | ----------------------------------------- |
| $\alpha$          | Learning rate                             |
| $\gamma$          | Discount factor                           |
| $\epsilon$        | Exploration rate (Q-learning)             |
| $\|\mathcal{B}\|$ | Batch size                                |
| $N$               | Trajectories per update (policy gradient) |

### Performance Metric

Win rate: $W(\theta) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[\text{P1 wins episode } m]$
```

Done. Simplified the adversary section to make clear it's a fixed truth-telling opponent (removed the $\pi_2^{TT}$ superscript notation since there's only one policy).

---

**User**

this shit is too complicated, our class talked about the gradiant descent and loss functions, but its in terms of T\*Q_theta - Q_theta >= 0 type beat i don't fully understand

adjust math2.md, and also within this chat and not in math2.md explain everything so i have background knowledge to verify your work

---

**Cursor**

Let me explain the fundamentals first, then simplify the doc.

---

## Background: Q-Learning

### The Bellman Equation

The optimal Q-function $Q^*$ satisfies:

$$Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')$$

This says: "the value of taking action $a$ in state $s$ equals the immediate reward $r$ plus the discounted best future value."

### The Bellman Operator $T$

The Bellman operator $T$ is a function that takes any Q-function and produces a new one:

$$(TQ)(s, a) = r + \gamma \max_{a'} Q(s', a')$$

The optimal $Q^*$ is the **fixed point**: $TQ^* = Q^*$

### Bellman Error

If we have an approximate $Q_\theta$, the **Bellman error** measures how far it is from satisfying the Bellman equation:

$$\delta = TQ_\theta - Q_\theta = \underbrace{(r + \gamma \max_{a'} Q_\theta(s', a'))}_{\text{target}} - \underbrace{Q_\theta(s, a)}_{\text{current estimate}}$$

This $\delta$ is also called the **TD error** (temporal difference error).

### Why $TQ_\theta - Q_\theta \geq 0$?

If $Q_\theta$ underestimates the true value, then $TQ_\theta > Q_\theta$ (positive error). The Bellman operator "pulls up" underestimates. At the fixed point, $TQ^* - Q^* = 0$.

### Q-Learning Update

We want to minimize Bellman error, so we do gradient descent:

$$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_\theta Q_\theta(s, a)$$

where $\delta = r + \gamma \max_{a'} Q_\theta(s', a') - Q_\theta(s, a)$.

This pushes $Q_\theta$ toward satisfying the Bellman equation.

---

## Background: Policy Gradient

### Objective

For stochastic policy $\pi_\theta(a|s)$, maximize expected return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

where $G(\tau) = \sum_{t=0}^T \gamma^t r_t$ is the trajectory return.

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot G\right]$$

The $\nabla_\theta \log \pi_\theta(a|s)$ term points in the direction that increases probability of action $a$. We weight it by $G$ (how good the outcome was).

### Baseline

To reduce variance, subtract a baseline $b$:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (G - b)\right]$$

Common choice: $b = \bar{G}$ (average return).

---

Now let me simplify MATH2.md to match this:

```markdown
## Liar's Dice w/ Reinforcement Learning

Core formulation ($S$, $A$, $T(s' \mid s, a)$, $\Omega(o \mid s', a)$, $R(s, a)$) defined in MATH.md.

### Adversary Policy (Fixed)

Opponent uses truth-telling policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$:

$\pi_2(o_2) = \begin{cases}
C & \text{if } \text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount} \\
\min\{a \in \text{LEGAL}(D_{\text{last}}) : \text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}\} & \text{if exists} \\
\min\{a \in \text{LEGAL}(D_{\text{last}})\} & \text{otherwise}
\end{cases}$

This fixed policy is incorporated into the MDP dynamics.

### Multi-Episode Training

$H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,2,3,4,5,6\}^N)$ for episode $m \in \{1, ..., M\}$

Trajectory: $\tau = (o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T, a_T, r_T)$

### Q-Learning (Value-Based)

**Function Approximation:**

$Q_\theta(o, a) = \theta^T \phi(o, a)$ where $\phi: O \times A \rightarrow \mathbb{R}^d$ is feature basis.

**Bellman Operator:**

$(T Q_\theta)(o, a) = r + \gamma \max_{a' \in \text{LEGAL}(o')} Q_\theta(o', a')$

**Bellman Error (TD Error):**

$\delta = T Q_\theta - Q_\theta = r + \gamma \max_{a'} Q_\theta(o', a') - Q_\theta(o, a)$

**Update Rule:**

$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_\theta Q_\theta(o, a)$

At convergence: $T Q_\theta = Q_\theta$ (fixed point), so $\delta = 0$.

**$\epsilon$-Greedy Exploration:**

$\pi_\epsilon(a|o) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\text{LEGAL}(o)|} & \text{if } a = \arg\max_{a'} Q_\theta(o, a') \\
\frac{\epsilon}{|\text{LEGAL}(o)|} & \text{otherwise}
\end{cases}$

### Policy Gradient (Policy-Based)

**Stochastic Policy:**

$\pi_\theta(a|o) = \text{softmax}(f_\theta(o, a))$ over $a \in \text{LEGAL}(o)$

**Trajectory Return:**

$G = \sum_{t=0}^T \gamma^t r_t$

**Objective:**

$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G]$

**Policy Gradient with Baseline:**

$\nabla_\theta J = \mathbb{E}\left[(G - b) \cdot \nabla_\theta \log \pi_\theta(a|o)\right]$

where $b = \bar{G}$ (mean return over batch).

**Update Rule:**

$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J$

### Hyperparameters

| Symbol     | Description                   |
| ---------- | ----------------------------- |
| $\alpha$   | Learning rate                 |
| $\gamma$   | Discount factor               |
| $\epsilon$ | Exploration rate (Q-learning) |

### Performance Metric

$W = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[\text{P1 wins episode } m]$
```

Done. Simplified to use Bellman operator notation. Key changes:

**Q-Learning section now shows:**

- Bellman operator $T$: applies the Bellman equation to any Q
- TD error $\delta = TQ_\theta - Q_\theta$: how far current Q is from satisfying Bellman
- Update: $\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_\theta Q_\theta$

**The intuition:**

- If $\delta > 0$: we underestimated, push Q up
- If $\delta < 0$: we overestimated, push Q down
- At convergence $\delta = 0$: Q satisfies Bellman equation

**Policy gradient section now shows:**

- Objective $J(\theta) = \mathbb{E}[G]$
- Gradient: $(G - b) \cdot \nabla_\theta \log \pi_\theta$
- Update: $\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J$

**The intuition:**

- $\nabla_\theta \log \pi_\theta(a|o)$ points toward making action $a$ more likely
- Weight by $(G - b)$: positive if better than average, negative if worse
- Good outcomes → increase probability; bad outcomes → decrease probability

---

**User**

currently the math2.md is a bit generic, can you make it specific to our liar's dice approach

---

**Cursor**

```markdown
## Liar's Dice w/ Reinforcement Learning

Core formulation ($S$, $O$, $A$, $T$, $\Omega$, $R$) defined in MATH.md.

### Adversary Policy (Fixed)

Opponent uses truth-telling policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$:

$\pi_2(o_2) = \begin{cases}
C & \text{if } \text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount} \\
\min\{a \in \text{LEGAL}(D_{\text{last}}) : \text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}\} & \text{if exists} \\
\min\{a \in \text{LEGAL}(D_{\text{last}})\} & \text{otherwise}
\end{cases}$

This fixed policy is incorporated into the MDP dynamics.

### Multi-Episode Training

Each episode $m$: $H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,...,6\}^N)$

Trajectory: $\tau = (o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T, a_T, r_T)$

where $o_t = (H_1, D_t, G_o)$ and $a_t \in \{B, C\}$ (bid or call LIAR).

Terminal reward only: $r_t = 0$ for $t < T$, $r_T = R(s_T, a_T) \in \{+1, -1\}$ (from MATH.md).

### Q-Learning (Value-Based)

**Feature Basis for Liar's Dice:**

$\phi(o, a) \in \mathbb{R}^d$ where $o = (H_1, D, G_o)$, $a \in \{B, C\}$:

- $\text{COUNT}(H_1, v)$ for $v \in \{1,...,6\}$ — dice counts in our hand
- $D_{\text{last}}.\text{Value}$, $D_{\text{last}}.\text{Amount}$ — last bid info (0 if $D = \emptyset$)
- $P_{valid}(D_{\text{last}} \mid H_1)$ — probability last bid is valid (from MATH.md)
- $a.\text{Value}$, $a.\text{Amount}$ if $a = B$; indicator $\mathbf{1}[a = C]$ if calling LIAR

**Q-Function:**

$Q_\theta(o, a) = \theta^T \phi(o, a)$

**Bellman Operator:**

$(T Q_\theta)(o, a) = r + \gamma \max_{a' \in \text{LEGAL}(D')} Q_\theta(o', a')$

where $o' = (H_1, D', G_o')$ is next observation after P1 plays $a$ and P2 responds via $\pi_2$.

**TD Error:**

$\delta = r + \gamma \max_{a' \in \text{LEGAL}(D')} Q_\theta(o', a') - Q_\theta(o, a)$

Note: if $a = C$ or P2 calls LIAR, game ends and $\max_{a'} Q_\theta(o', a') = 0$.

**Update:**

$\theta \leftarrow \theta + \alpha \cdot \delta \cdot \nabla_\theta Q_\theta(o, a)$

At convergence: $\delta = 0$ (Q satisfies Bellman equation).

**$\epsilon$-Greedy Exploration:**

$\pi_\epsilon(a|o) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\text{LEGAL}(D)|} & \text{if } a = \arg\max_{a'} Q_\theta(o, a') \\
\frac{\epsilon}{|\text{LEGAL}(D)|} & \text{otherwise}
\end{cases}$

where $\text{LEGAL}(D) \subset \{B, C\}$ depends on bid history (from MATH.md).

### Policy Gradient (Policy-Based)

**Stochastic Policy:**

$\pi_\theta(a|o) = \frac{\exp(f_\theta(\phi(o, a)))}{\sum_{a' \in \text{LEGAL}(D)} \exp(f_\theta(\phi(o, a')))}$

**Trajectory Return:**

$G = \sum_{t=0}^T \gamma^t r_t = \gamma^T r_T \in \{+\gamma^T, -\gamma^T\}$

(Only terminal reward matters; earlier $r_t = 0$.)

**Objective:**

$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G]$

**Policy Gradient with Baseline:**

$\nabla_\theta J = \mathbb{E}\left[(G - b) \cdot \nabla_\theta \log \pi_\theta(a|o)\right]$

where $b = \bar{G}$ (mean return over batch of games).

**Update:**

$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J$

### Hyperparameters

| Symbol     | Description                   |
| ---------- | ----------------------------- |
| $\alpha$   | Learning rate                 |
| $\gamma$   | Discount factor               |
| $\epsilon$ | Exploration rate (Q-learning) |
| $N$        | Dice per player               |

### Performance Metric

$W = \frac{1}{M} \sum_{m=1}^M \mathbf{1}[\text{P1 wins game } m]$
```

Done. Now Liar's Dice specific:

**Feature basis** — uses our game's variables:

- Dice counts from $H_1$
- Last bid value/amount from $D_{\text{last}}$
- $P_{valid}$ from MATH.md
- Action encoding (bid value/amount or LIAR indicator)

**Bellman operator** — shows that $o' = (H_1, D', G_o')$ comes after P1 plays and P2 responds via truth-telling policy

**Terminal reward** — explicit that $r_t = 0$ during game, only $r_T \in \{+1, -1\}$ at end

**Legal actions** — references $\text{LEGAL}(D)$ from MATH.md (depends on bid history)

**Return** — simplified to $G = \gamma^T r_T$ since only terminal reward matters
