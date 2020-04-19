from gym.envs.registration import register
from andes_gym.envs.andes_freq import AndesFreqControl


register(
    id='AndesFreqControl-v0',
    entry_point='andes_gym:AndesFreqControl',
)


