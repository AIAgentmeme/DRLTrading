import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from env.EnvMultipleCrypto_train import CryptoEnvTrain
from env.EnvMultipleCrypto_validation import CryptoEnvValidation

class CryptoModelRunner:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.data = None

    def load_dataset(self):
        if os.path.exists(self.file_name):
            self.data = pd.read_csv(self.file_name, index_col=0)
        else:
            print(f"File {self.file_name} not found.")

    def train_model(self, model_type, env_train, model_name, timesteps=25000):
        start = time.time()

        if model_type == "A2C":
            model = A2C('MlpPolicy', env_train, verbose=0)
        elif model_type == "DDPG":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env_train.action_space.shape),
                                                        sigma=0.1 * np.ones(env_train.action_space.shape))
            model = DDPG('MlpPolicy', env_train, action_noise=action_noise)
        elif model_type == "PPO":
            model = PPO('MlpPolicy', env_train, ent_coef=0.005, batch_size=64, n_steps=2048)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.learn(total_timesteps=timesteps)
        end = time.time()

        model.save(f"trained_models/{model_name}")
        print(f"Training time ({model_type}):", (end - start) / 60, 'minutes')

        return model

    def run_model(self):
        if self.data is None:
            self.load_dataset()

        # Ensure data is preprocessed
        if self.data is None:
            raise RuntimeError("Data not loaded. Please ensure the dataset is preprocessed and available.")

        # Define unique trade timestamps
        unique_trade_timestamps = sorted(set(self.data.timestamp))
        print("Unique Trade Timestamps:", unique_trade_timestamps[:10])

        rebalance_window = 24 * 7
        validation_window = 24 * 7

        # Run ensemble strategy
        self.run_ensemble_strategy(unique_trade_timestamps, rebalance_window, validation_window)

    def get_unique_trade_timestamps(self, start_timestamp: int, end_timestamp: int) -> np.ndarray:
        return self.data[(self.data.timestamp > start_timestamp) & (self.data.timestamp <= end_timestamp)].timestamp.unique()

    def run_ensemble_strategy(self, unique_trade_timestamps, rebalance_window, validation_window):
        print("============Start Ensemble Strategy============")

        last_state_ensemble = []
        ppo_sharpe_list, ddpg_sharpe_list, a2c_sharpe_list, model_use = [], [], [], []
        
        start_timestamp = int(datetime.strptime("2022-01-05", "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime("2024-01-04", "%Y-%m-%d").timestamp())

        insample_turbulence = self.data[(self.data.timestamp < end_timestamp) & (self.data.timestamp >= start_timestamp)]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=['timestamp'])
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

        start = time.time()

        # The for loop moves 1 week at a time
        for i in range(rebalance_window + validation_window, len(unique_trade_timestamps), rebalance_window):
            print("============================================")

            initial = i - rebalance_window - validation_window == 0

            validation_start_timestamp = unique_trade_timestamps[i - rebalance_window - validation_window]
            end_timestamp_index = (self.data.index[self.data["timestamp"] == validation_start_timestamp].to_list()[-1])
            start_timestamp_index = end_timestamp_index - validation_window * 24 + 1
            historical_turbulence = (self.data.iloc[start_timestamp_index:(end_timestamp_index + 1), :].drop_duplicates(subset=['timestamp']))
            historical_turbulence_mean = historical_turbulence.turbulence.mean()

            if historical_turbulence_mean > insample_turbulence_threshold:
                turbulence_threshold = insample_turbulence_threshold
            else:
                turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

            print("turbulence_threshold:", turbulence_threshold)

            start_timestamp = datetime.strptime("2022-01-05", "%Y-%m-%d").timestamp()
            train = self.data[(self.data.timestamp >= start_timestamp) & (self.data.timestamp < unique_trade_timestamps[i - rebalance_window - validation_window])]
            if train.empty:
                print("Train is empty")
                continue
            env_train = DummyVecEnv([lambda: CryptoEnvTrain(train)])

            validation = self.data[(self.data.timestamp >= unique_trade_timestamps[i - rebalance_window - validation_window]) &
                                    (self.data.timestamp < unique_trade_timestamps[i - rebalance_window])]
            env_val = DummyVecEnv([lambda: CryptoEnvValidation(validation, turbulence_threshold=turbulence_threshold, iteration=i)])
            print(type(env_val))  # Check the type of env_val
            # obs_val, info = env_val.reset()
            obs_val = env_val.reset()

            print("======A2C Training========")
            model_a2c = self.train_model("A2C", env_train, model_name=f"A2C_{i}", timesteps=100000)
            print("======PPO Training========")
            model_ppo = self.train_model("PPO", env_train, model_name=f"PPO_{i}", timesteps=100000)
            print("======DDPG Training========")
            model_ddpg = self.train_model("DDPG", env_train, model_name=f"DDPG_{i}", timesteps=100000)

        end = time.time()
        print("Ensemble Strategy took:", (end - start) / 60, "minutes")

if __name__ == "__main__":
    file_name = "data/processed_crypto_data_with_indicators.csv"
    runner = CryptoModelRunner(file_name)
    runner.run_model()

