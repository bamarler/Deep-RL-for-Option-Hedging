from option_gym import OptionEnv

env = OptionEnv()

# Run an episode until it ends :
done, truncated = False, False
obs, info = env.reset()
while not done and not truncated:
    print(obs)
    # Pick a position by its index in your position list (=[-1, 0, 1])....usually something like : position_index = your_policy(observation)
    position_index = env.action_space.sample() # At every timestep, pick a random position index from your position list (=[-1, 0, 1])
    obs, reward, done, truncated, info = env.step(position_index)