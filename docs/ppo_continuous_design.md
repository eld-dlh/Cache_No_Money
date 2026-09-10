# PPO for Continuous Frequency Tuning & Multi-Receiver Interception

## Step 12 — Design Document (Person 3)

> [!NOTE]
> This is a **design document only** — no implementation is required for the current milestone.
> It outlines how the system would be extended beyond discrete DQN to handle continuous
> frequency tuning and multi-receiver coordination.

---

## 1. Motivation: Why Move Beyond Discrete DQN?

The current Tier 2 DRQN operates on **64 discrete frequency channels**, each covering a 281.25 MHz band. This works well for the current simulation but has two fundamental limitations:

### Limitation 1: Discrete Channels Miss Fine-Grained Emitters
- A radar emitter transmitting at exactly 5,100 MHz falls in the boundary between channels 18 and 19.
- The agent must pick one channel — it can't tune to exactly 5,100 MHz.
- In real SDR hardware, receivers can tune to arbitrary frequencies with kHz precision.
- **Solution:** Output a continuous frequency $f \in [f_{\min}, f_{\max}]$ instead of a discrete channel index.

### Limitation 2: Single Receiver Can't Cover Multiple Emitters
- The current system has one receiver scanning one channel per time step.
- If 3 emitters are active simultaneously on channels 12, 30, and 55, we can only intercept one.
- Modern cognitive radar platforms have $M$ parallel receivers (e.g., $M = 4$).
- **Solution:** Coordinate $M$ receivers to simultaneously cover different bands.

---

## 2. Proposed Architecture: Recurrent PPO

### Why PPO Instead of DQN?

| Feature | DQN | PPO |
|---------|-----|-----|
| Action space | Discrete only | Continuous or discrete |
| Policy type | Implicit (argmax Q) | Explicit (parameterised distribution) |
| Stability | Sensitive to hyperparameters | Clipped objective prevents large updates |
| Multi-output | Complex (combinatorial explosion) | Natural (multi-dim Gaussian) |

PPO (Proximal Policy Optimisation) is the standard choice for continuous action spaces because it directly parameterises a probability distribution over actions.

### Network Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Input: normalised pulse window (batch, seq_len, 5)           │
│        + context features (batch, D_eng)                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │ Feature Extract  │  Linear(5 → 128) + LayerNorm + ReLU
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   2-Layer LSTM   │  LSTM(128, 256)
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
  ┌────────▼────────┐     ┌────────▼────────┐
  │   Actor Head    │     │   Critic Head    │
  │ (Policy π_θ)    │     │ (Value V_θ)      │
  └────────┬────────┘     └────────┬────────┘
           │                       │
  ┌────────▼────────┐     ┌────────▼────────┐
  │ μ ∈ ℝ^M         │     │ V(s) ∈ ℝ        │
  │ σ ∈ ℝ^M (>0)    │     │                  │
  └─────────────────┘     └──────────────────┘
```

**Actor Head (Policy):**
- Outputs parameters of a **Gaussian distribution** for each of $M$ receivers:
  - $\mu_i \in [f_{\min}, f_{\max}]$ — mean tuning frequency for receiver $i$
  - $\sigma_i > 0$ — uncertainty (exploration width) for receiver $i$
- Action sampling: $a_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, clamped to valid range.
- During evaluation: $a_i = \mu_i$ (deterministic).

**Critic Head (Value Baseline):**
- Outputs a scalar $V(s)$ estimating the expected return from state $s$.
- Used to compute the advantage $\hat{A}_t = R_t - V(s_t)$ for variance reduction.

---

## 3. PPO Clipped Objective

The core PPO loss prevents catastrophically large policy updates:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ is the probability ratio
- $\hat{A}_t$ is the generalised advantage estimate (GAE)
- $\epsilon = 0.2$ is the clipping parameter

### Full Loss Function

$$L(\theta) = -L^{\text{CLIP}}(\theta) + c_1 \cdot L^{\text{VF}}(\theta) - c_2 \cdot H[\pi_\theta]$$

| Term | Purpose | Default Weight |
|------|---------|----------------|
| $L^{\text{CLIP}}$ | Policy improvement | 1.0 |
| $L^{\text{VF}}$ | Value function accuracy (MSE) | $c_1 = 0.5$ |
| $H[\pi_\theta]$ | Entropy bonus (encourages exploration) | $c_2 = 0.01$ |

---

## 4. Multi-Receiver Coordination

### Option A: Independent Learners (Simplest)
- Each of $M$ receivers has its own PPO agent.
- Agents share the same observation but select frequencies independently.
- **Pro:** Simple. **Con:** Receivers may redundantly tune to the same frequency.

### Option B: Joint Action Space (Centralised)
- Single policy outputs $M$ frequency values simultaneously: $\mathbf{a} \in \mathbb{R}^M$.
- A **diversity penalty** is added to the reward to discourage overlap:
  $$R_{\text{diversity}} = -\lambda \sum_{i \neq j} \max(0, \Delta f_{\text{min}} - |f_i - f_j|)$$
- **Pro:** Explicit coordination. **Con:** Action space grows with $M$.

### Option C: CTDE — Centralized Training, Decentralized Execution (Recommended)
- **Training:** A central critic observes all $M$ receivers' states and actions to compute a shared value estimate.
- **Execution:** Each receiver runs its own lightweight actor independently (no communication needed at inference time).
- **Architecture:**
  ```
  Training:   Shared Critic V(s, a_1, ..., a_M)
  Inference:  Actor_i(s) → a_i    (runs independently per receiver)
  ```
- **Pro:** Best of both worlds — coordinated learning, independent deployment.
- **Con:** More complex training setup.

---

## 5. Continuous Reward Adaptation

The current reward function uses discrete channel matching:
```
reward = +1.0 if chosen_channel == pulse_channel
```

For continuous frequencies, this becomes a **proximity-based reward**:

$$R_{\text{intercept}} = \exp\left(-\frac{(f_{\text{tuned}} - f_{\text{pulse}})^2}{2 \cdot \text{BW}^2}\right)$$

Where $\text{BW}$ is the receiver bandwidth (e.g., 5 MHz). This gives:
- $R = 1.0$ when perfectly tuned ($f_{\text{tuned}} = f_{\text{pulse}}$)
- $R \approx 0.6$ when 1 bandwidth away
- $R \approx 0.0$ when far off-frequency

---

## 6. Estimated Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | $3 \times 10^{-4}$ | Standard PPO LR |
| Clip $\epsilon$ | 0.2 | Standard clipping range |
| GAE $\lambda$ | 0.95 | Balance bias-variance in advantage |
| Discount $\gamma$ | 0.99 | Same as DRQN |
| Mini-batch size | 64 | Fits in GPU memory |
| PPO epochs per rollout | 4 | Multiple passes over collected data |
| Entropy coefficient $c_2$ | 0.01 | Encourage exploration |
| Value loss coefficient $c_1$ | 0.5 | Standard weight |
| Max gradient norm | 0.5 | PPO typically uses tighter clipping |
| $M$ (receivers) | 1–4 | Start with 1, scale to 4 |

---

## 7. Migration Path from Current DRQN

The transition from the current discrete DRQN to continuous PPO would follow these steps:

1. **Phase 1:** Keep 64 discrete channels but replace DQN with PPO (discrete PPO with categorical policy). This validates the PPO training infrastructure without changing the action space.

2. **Phase 2:** Switch to continuous frequency output ($M = 1$ receiver). Replace the categorical policy with a Gaussian policy. Adapt the reward function to proximity-based.

3. **Phase 3:** Scale to $M > 1$ receivers using CTDE. Add the diversity penalty.

4. **Phase 4:** Integrate with Person 4's real SDR hardware. Validate latency budget ($\le 2$ ms per receiver).

---

## 8. References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017). arXiv:1707.06347
- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017). arXiv:1706.02275 (CTDE framework)
- Haarnoja et al., "Soft Actor-Critic" (2018). arXiv:1801.01290 (alternative to PPO for continuous control)
