import pickle
from stable_baselines3.common.callbacks import BaseCallback

class SaveTrainingStatsCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveTrainingStatsCallback, self).__init__(verbose)
        self.save_path = save_path
        self.rewards = []
        self.episode_lengths = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.rewards.append(self.locals["infos"][i].get("episode", {}).get("r"))
                    self.episode_lengths.append(self.locals["infos"][i].get("episode", {}).get("l"))
                    self.timesteps.append(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        data = {
            "rewards": self.rewards,
            "episode_lengths": self.episode_lengths,
            "timesteps": self.timesteps,
        }
        with open(self.save_path, "wb") as f:
            pickle.dump(data, f)
