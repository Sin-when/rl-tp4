# RL-TP4

## Implementation Choices

When training our DQN agent for Pong, we encountered a significant challenge: the original Pong reward structure only provides:
  - +1 for scoring a point,
  - -1 for losing a point,
  - and 0 for most other actions.

At the start, with an epsilon value of 1, the agent begins by taking completely random actions. This setup makes it unlikely to earn positive rewards, leading to a slow learning process as the agent struggles to associate actions with positive outcomes.

### Solution: Reward Shaping

To address these issues, we implemented *reward shaping* to provide more immediate feedback:

1. **Small positive reward for hitting the ball**: This provides immediate positive reinforcement for a crucial game action and helps the agent quickly learn the importance of intercepting the ball.

2. **Small negative reward for missing the ball**: This discourages poor paddle positioning, enabling the agent to learn from mistakes without waiting for a point loss.

3. **Reward based on the paddle’s distance to the ball**: This continuous feedback on positioning helps the agent keep the paddle close to the ball, improving its chances of hitting the ball consistently.

With these new implementations, the agent receives meaningful feedback on almost every frame, rather than only when points are scored. This approach enables the agent to learn useful strategies much earlier in the training process.

Once the agent understands that it should prioritize hitting the ball, we can revert to the original Pong reward system, based solely on points. By then, the agent will have developed foundational skills, allowing it to focus on scoring points and use its strategies more effectively.
