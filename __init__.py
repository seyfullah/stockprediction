import gym

gym.register(
    id='StockEnvironment-v0',
    entry_point='StockEnvironment',
    max_episode_steps=200,
)