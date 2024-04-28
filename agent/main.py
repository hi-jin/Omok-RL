import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np
import argparse

try:
    from omok_env import OmokEnv
except ImportError:
    from agent.omok_env import OmokEnv

try:
    from omok_feature_extractor import OmokFeatureExtractor
except ImportError:
    from agent.omok_feature_extractor import OmokFeatureExtractor

from omok.omok import Omok, StateAfterPutStone, Stone


class UpdateOpponentCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(UpdateOpponentCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        global env_without_wrapper, black
        # 매 check_freq 타임스텝마다 실행될 로직
        if self.n_calls % self.check_freq == 0:
            black.save("black")
            env_without_wrapper.copied_agent = sb3.PPO.load("black")  # type: ignore
        return True


def main(args):
    if args.logging:
        run = wandb.init(
            project=args.project,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    omok = Omok()
    env_without_wrapper = OmokEnv(omok, Stone.BLACK, render_mode="rgb_array")
    env = DummyVecEnv([lambda: Monitor(env_without_wrapper)])
    if args.logging:
        env = VecVideoRecorder(
            env,  # type: ignore
            f"videos/{run.id}",  # type: ignore
            record_video_trigger=lambda x: x % 10000 == 0,
            video_length=300,
        )
    black = sb3.PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(
            features_extractor_class=OmokFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
            ),
        ),
        tensorboard_log=f"runs/{run.id}" if args.logging else None,
    )
    env_without_wrapper.main_agent = black  # type: ignore

    black.save("black")
    env_without_wrapper.copied_agent = sb3.PPO.load("black")  # type: ignore

    callbacks = [UpdateOpponentCallback(check_freq=10000)]
    if args.logging:
        callbacks.append(
            WandbCallback(
                model_save_path="./models/",
                model_save_freq=10000,
                verbose=2,
            )
        )

    black.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    raise argparse.ArgumentTypeError("Unsupported type for boolean value.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging", type=str2bool, default=True)
    parser.add_argument("--project", type=str, default="omok")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
