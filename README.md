# Variations of the Multi-Armed Bandit Problem: a Reinforcement Learning Project

## Background
The multi-armed bandit reinforcement learning problem is where an agent is tasked to **maximize its total rewards** over many arm pulls. Each arm has different probabilities for giving out a reward for each pull, in which the agent is initially unaware of. Each time an agent pulls an arm, it keeps an estimate of that arm's probability and updates it from the wins and losses. 
  There are two concepts emphasized in the multi-armed bandit problem: **exploitation**, where the agent taking advantage of the arm that it currently estimates for it to have the highest probability of winning, and **exploration**, where the agent tries out different arm to see if they have a higher probability of winning. The agent **cannot do exploitation and exploration simultaneously**, as exploitation risks locking onto a mediocre arm and missing out on a better arm, and exploration means pulling an arm that may be worse and losing some rewards. 

This project uses an epsilon-greedy agent with probability 0.1 it picks a random arm, otherwise it picks the arm with its highest estimated win rate Q. The incremental update is as follows:
```
Q = Q + step * (reward - Q)
```

Which translates to: 
```
n = n + 1
Q(a) <- Q(a) + 1/n * (r - Q(a))
```

Where:
- n = number of times action "a" was performed
- Q(a) = value estimate of action "a"
- r(a) = reward of sampling action bandit (bandit) "a"

# The Non-Stationary Multi-Armed Bandit
How can an agent adapt to a changing environment when it can only learn by trial an error? In this case, relying on old data from several trials ago may not necessarily be beneficial compared to a consistent, steady environment. To simulate this, I ran an agent in a steady world, where the arms' win rates never change, and another in a changing world.

## Setup
The only difference between the two agents is the "step" portion. The agent that takes account of every single result equally (average) has a step of 1/n. The agent that focuses on the recent results (fixed)  has a constant step size. In other words, the fixed step size represents an agent coping with a changing environment by eventually "forgetting" the outcomes from previous actions. 

I then ran both agents in the steady world and in the changing world. The agent never sees the true win rates. It only sees wins and losses, so it has to notice the change from its own results.

## Results
In the steady world, the average agent is ahead throughout, whereas in the changing world, the fixed agent eventually took the lead. 
The graph of the results can be viewed in steady_world.png and changing_world.png
Additionally, the results came from one seed, so it is worth rechecking with other seed sizes. Through these results, I learned that a plain average assumes the world holds still. When the world changes, old data becomes misleading and the average is slow to adjust. A fixed step size trades accuracy for adaptability, as forgets stale results, but it is jittery even when nothing changes. Furthermore, averaging over many games matters. A single game is mostly luck, and small gaps between curves are hard to trust when there are few games.

### Sensitivity to change size
| Change (std dev) | Average (1/n) | Fixed step |
|---|---|---|
| 0 | [ 0.74607 ] | [ 0.71003 ] |
| 0.01 | [ 0.76979 ] | [ 0.78079 ] |
| 0.03 | [ 0.67995 ] | [ 0.75774 ] |
| 0.05 | [ 0.6308 ] | [ 0.72635 ] |
| 0.1 | [ 0.58763 ] | [ 0.67762 ] |

The fixed step overtakes the 1/n step when the change = 0.01 

### Sensitivity to step size
| Alpha | Avg reward (change = 0.01) |
|---|---|
| 0.01 | [ 0.50248 ] |
| 0.05 | [ 0.59377 ] |
| 0.1 | [ 0.66383 ] |
| 0.3 | [ 0.72806 ] |
| 0.5 | [ 0.72141 ] |

Best alpha at change: 0.3

## How to run
Run
```
pip install numpy matplotlib
python multi_armed_bandit.py
```
The plots are saved to the output folder. 

## Limitations and next steps
The win rates and change are randomly generated numbers in a simulation, which showcase the learning mechanics but are not necessarily solutions to a real-world problem. The next step is to add a UCB agent, track regret, and extend to a contextual bandit where the best arm is dependent on what the agent observes. 

# Acknowledgements
This project builds on the epsilon-greedy multi-armed bandit by Anson Wong (ankonzoid): https://github.com/ankonzoid/LearningX/blob/master/classical_RL/multiarmed_bandit/multiarmed_bandit.py

The baseline Environment and Agent structure comes from that code. The changing environment, the fixed step agent, the two-world comparison and the write-up are my additions.




