from setuptools import setup

setup(
    name='gym_simple_minigrid',
    version='2.0.0',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/rafelps/gym-simple-minigrid',
    description='Simple gridworld package for OpenAI Gym with multi-goal support',
    packages=['gym_simple_minigrid', 'gym_simple_minigrid.envs'],
    install_requires=['gym', 'numpy', 'matplotlib']
)
