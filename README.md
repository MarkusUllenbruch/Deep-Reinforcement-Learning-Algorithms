# Deep Reinforcement-Learning
## Implementations of State-of-the-Art Deep Reinforcement-Learning Algorithms
 
Own Tensorflow 2.0 & PyTorch implementations of:

## Soft Actor-Critic (SAC)
- [SAC Paper 1](https://arxiv.org/abs/1801.01290)
  provides the original and first version of SAC.
- [SAC Paper 2](https://arxiv.org/abs/1812.05905)
  provides the State-of-the-Art implementation of SAC with automatic temperature parameter optimization.
  
  [(SAC Tensorflow Code)](SAC/agent.py)
  
  [(Complete Code as Jupyter Notebook)](SAC/SAC_TF.ipynb)
  
  ![alt text](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)


## Proximal Policy Optimization (PPO) continuous version
[PPO Paper](https://arxiv.org/abs/1707.06347)

[(PPO Tensorflow Code)](PPO/Tensorflow/agent.py) or [(PPO PyTorch Code)](PPO/PyTorch/agent.py)

![alt text](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)

## Additional Info
These algorithms provide state of the art performance in most openAI and Unity environments and are currently a big field of research in industry and research institutions like openAI and Google DeepMind.
The algorithms are implemented to master openAI environments, bu you dan easily adapt them to any other environment you want.

## To-Come:
### Deep Deterministic Policy Gradient (DDPG)
[DDPG Paper](https://arxiv.org/abs/1509.02971)

### Twin Delayed Deep Deterministic Policy Gradient (TD3)
[TD3 Paper](https://arxiv.org/abs/1802.09477)
