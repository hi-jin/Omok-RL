from env import OmokEnv
from typing import Literal
import multiprocessing


def play(role: Literal["black", "white"]):
    env = OmokEnv(8000, role)
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    with multiprocessing.Pool(2) as p:
        p.map(play, ["black", "white"])
