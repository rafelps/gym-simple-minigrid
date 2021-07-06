from gym.envs.registration import register as gym_register

env_list = []


def register(_id, entry_point, reward_threshold=0.95):
    assert _id.startswith("Simple-MiniGrid-")
    assert _id not in env_list

    # Register the environment with OpenAI gym
    gym_register(id=_id, entry_point=entry_point, reward_threshold=reward_threshold)

    # Add the environment to the set
    env_list.append(_id)
