
## Rewards
The total reward is: ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost*.

- *healthy_reward*:
Every timestep that the Walker2d is alive, it receives a fixed reward of value `healthy_reward`,
- *forward_reward*:
A reward for moving forward,
this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
$w_{forward} \times \frac{dx}{dt}$, where
$dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
$dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
and `frametime` which is 0.002 - so the default is $dt = 4 \times 0.002 = 0.008$,
$w_{forward}$ is the `forward_reward_weight` (default is $1$).
- *ctrl_cost*:
A negative reward to penalize the Walker2d for taking actions that are too large.
$w_{control} \times \\|action\\|_2^2$,
where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).

`info` contains the individual reward terms.