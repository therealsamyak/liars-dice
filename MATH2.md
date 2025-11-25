## Liar's Dice w/ Reinforcement Learning

Core formulation ($S$, $O$, $A$, $T$, $\Omega$, $R$) defined earlier.

### Opponent Policy (hard-coded)

Opponent uses truth-telling policy $\pi_2: O_2 \rightarrow A$ where $O_2 = (H_2, D, G_o)$:

$\pi_2(o_2) = \begin{cases}
C & \text{COUNT}(H_2, D_{\text{last}}.\text{Value}) + N < D_{\text{last}}.\text{Amount} \\
\min\{a \in \text{LEGAL}(D_{\text{last}}) : \text{COUNT}(H_2, a.\text{Value}) + N \ge a.\text{Amount}\} & \text{if exists} \\
\min\{a \in \text{LEGAL}(D_{\text{last}})\} & \text{otherwise}
\end{cases}$

Where $C = \text{Call "LIAR!"}$ as defined earlier.

This fixed policy is incorporated into the MDP dynamics.

### Multi-Episode Training

Each episode $m$: $H_1^{(m)}, H_2^{(m)} \sim \text{Uniform}(\{1,...,6\}^N)$

Trajectory: $\tau = (o_0, a_0, r_0, o_1, a_1, r_1, ..., o_T, a_T, r_T)$

where $o_t = (H_1, D_t, G_o)$ and $a_t \in \{B, C\}$ (bid or call LIAR).

Terminal reward only: $r_t = 0$ for $t < T$, $r_T = R(s_T, a_T) \in \{+1, -1\}$ (from earlier).

### Q-Learning (Value-Based)

**Feature Basis for Liar's Dice:**

$\phi(o, a) \in \mathbb{R}^d$ where $o = (H_1, D, G_o)$, $a \in \{B, C\}$:

- $\text{COUNT}(H_1, v)$ for $v \in \{1,...,6\}$ — dice counts in our hand
- $D_{\text{last}}.\text{Value}$, $D_{\text{last}}.\text{Amount}$ — last bid info (0 if $D = \emptyset$)
- $P_{valid}(D_{\text{last}} \mid H_1)$ — probability last bid is valid (from earlier)
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

where $\text{LEGAL}(D) \subset \{B, C\}$ depends on bid history (from earlier).

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
