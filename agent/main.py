import stable_baselines3 as sb3
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

try:
    from omok_env import OmokEnv
except ImportError:
    from agent.omok_env import OmokEnv
from omok.omok import Omok, StateAfterPutStone, Stone


wandb.init(
    project="omok",
)


omok = Omok()
env = OmokEnv(omok, Stone.BLACK)
black = sb3.PPO("MultiInputPolicy", env, verbose=1)
env.main_agent = black

black.save("black")
env.copied_agent = sb3.PPO.load("black")


from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        global env, black
        # 매 check_freq 타임스텝마다 실행될 로직
        if self.n_calls % self.check_freq == 0:
            black.save("black")
            env.copied_agent = sb3.PPO.load("black")
        return True


black.learn(
    500000,
    callback=[
        WandbCallback(),
        CustomCallback(check_freq=10000),
    ],
)
