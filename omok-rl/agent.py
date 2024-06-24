import stable_baselines3 as sb3
from env import OmokEnv
from argparse import ArgumentParser
from typing import Literal
import wandb
from wandb.integration.sb3 import WandbCallback


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--server-port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--who",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    run = wandb.init(
        project="omok-rl",
        name=f"agent_{args.who}",
        monitor_gym=True,
        save_code=True,
        sync_tensorboard=True,
    )

    env = OmokEnv(
        server_port=args.server_port,
        who=args.who,
    )

    agent = sb3.PPO(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=f"./runs/{run.id}",
        ent_coef=0.01,
    )
    agent.learn(
        total_timesteps=1050000,
        callback=WandbCallback(
            model_save_path=f"./models/{run.id}",
            model_save_freq=100000,
            verbose=2,
        ),
    )
    run.finish()
