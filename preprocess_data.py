import numpy as np
import pandas as pd
import sys
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt

class CryptoDataProcessor:
    def __init__(self, file_name: str):
        """
        Initialize the CryptoDataProcessor class with the dataset file name.
        :param file_name: (str) path to the dataset file.
        """
        self.file_name = file_name
        self.data = None

    def load_dataset(self):
        """
        Load CSV dataset from the given file path.
        """
        self.data = pd.read_csv(self.file_name)
        self.data.set_index(['index'], inplace=True)

    def data_split(self, start: int, end: int) -> pd.DataFrame:
        """
        Split the dataset into training or testing using the specified date range.
        :param start: (int) the start date.
        :param end: (int) the end date.
        :return: (pd.DataFrame) the filtered dataset.
        """
        return self.data[(self.data.date >= start) & (self.data.date < end)].sort_values(['date', 'asset'], ignore_index=True)

    def add_technical_indicators(self):
        """
        Add technical indicators to the dataset using the stockstats package.
        """
        # Initialize empty lists to store indicator values for all assets
        macd_list, rsi_list, cci_list, adx_list = [], [], [], []

        # Process each asset group separately
        for _, group in self.data.groupby('asset'):
            # Convert the group DataFrame to a StockDataFrame
            token = Sdf.retype(group.copy().reset_index())
            token.set_index('index', inplace=True)
 
            # Calculate technical indicators for this group
            macd_list.append(token['macd'])
            rsi_list.append(token['rsi_30'])
            cci_list.append(token['cci_30'])
            adx_list.append(token['dx_30'])

        # Concatenate the results back into the original DataFrame
        macd_df = pd.concat(macd_list).sort_index()
        rsi_df = pd.concat(rsi_list).sort_index()
        cci_df = pd.concat(cci_list).sort_index()
        adx_df = pd.concat(adx_list).sort_index()

        # Add the new columns to the original DataFrame
        self.data = self.data.join(macd_df, rsuffix='_macd')
        self.data = self.data.join(rsi_df, rsuffix='_rsi')
        self.data = self.data.join(cci_df, rsuffix='_cci')
        self.data = self.data.join(adx_df, rsuffix='_adx')

    def preprocess_data(self):
        """
        Data preprocessing pipeline: load, clean, and add technical indicators.
        """
        self.load_dataset()
        # self.data = self.data[self.data.date >= 20090000]  # Filter data after 2009
        self.add_technical_indicators()
        self.data.bfill(inplace=True)  # Fill missing values using backward fill
        self.add_turbulence()

    def calculate_turbulence(self) -> pd.DataFrame:
        """
        Calculate the turbulence index for the dataset.
        """
        df_price_pivot = self.data.pivot(index='timestamp', columns='asset', values='close')
        unique_timestamps = self.data.timestamp.unique()
        
        print(df_price_pivot.head())

        # Initialize turbulence index list with 0s for the first year (252 days)
        turbulence_index = [0] * 240
        for i in range(240, len(unique_timestamps)):
            current_price = df_price_pivot.loc[unique_timestamps[i]]
            hist_price = df_price_pivot.loc[unique_timestamps[:i]]

            # Calculate covariance and turbulence index
            cov_temp = hist_price.cov()
            current_temp = (current_price - np.mean(hist_price, axis=0)).values
            temp = current_temp.dot(np.linalg.inv(cov_temp)).dot(current_temp.T)

            # Avoid large outliers at the beginning of the data
            turbulence_temp = temp if temp > 0 else 0
            turbulence_index.append(turbulence_temp)

        return pd.DataFrame({'timestamp': df_price_pivot.index, 'turbulence': turbulence_index})

    def add_turbulence(self):
        """
        Add turbulence index to the dataset.
        """
        turbulence_index = self.calculate_turbulence()
        self.data = self.data.merge(turbulence_index, on='timestamp').sort_values(['timestamp', 'asset']).reset_index(drop=True)

    def get_date_range(self):
        """
        Get the date range of the dataset.
        """
        return self.data.date.min(), self.data.date.max()

    def summary_statistics(self):
        """
        Display summary statistics for the dataset.
        """
        if self.data is not None:
            print("Summary Statistics:\n")
            print(self.data.describe())
        else:
            print("Data has not been loaded or processed yet.")

    def plot_technical_indicators(self):
        """
        Plot the technical indicators for three random assets for the first 100 datapoints.
        """
        if self.data is None:
            print("Data has not been processed yet. Please preprocess the data first.")
            return

        # Select three random assets
        random_assets = self.data['asset'].drop_duplicates().sample(3).values

        for asset in random_assets:
            asset_data = self.data[self.data['asset'] == asset].iloc[:100]

            plt.figure(figsize=(12, 8))

            # Plot close price
            plt.subplot(4, 1, 1)
            plt.plot(asset_data['timestamp'], asset_data['close'], label='Close Price')
            plt.title(f'{asset} - Close Price')
            plt.legend()

            # Plot MACD
            plt.subplot(4, 1, 2)
            plt.plot(asset_data['timestamp'], asset_data['macd'], label='MACD', color='orange')
            plt.title(f'{asset} - MACD')
            plt.legend()

            # Plot RSI
            plt.subplot(4, 1, 3)
            plt.plot(asset_data['timestamp'], asset_data['rsi_30'], label='RSI', color='green')
            plt.title(f'{asset} - RSI')
            plt.legend()

            # Plot CCI
            plt.subplot(4, 1, 4)
            plt.plot(asset_data['timestamp'], asset_data['cci_30'], label='CCI', color='red')
            plt.title(f'{asset} - CCI')
            plt.legend()

            plt.tight_layout()
            plt.show()

    def save_dataset(self, output_file_name: str):
        """
        Save the processed dataset to the hard drive.
        :param output_file_name: (str) Path to save the dataset.
        """
        self.data.columns = ['date', 'hour', 'asset', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'cci', 'adx', 'turbulence']
        if self.data is not None:
            self.data.to_csv(output_file_name, index=True)
            print(f"Dataset saved to {output_file_name}")
        else:
            print("Data has not been processed yet. Cannot save.")

if __name__ == "__main__":
    file_name = "data/processed_crypto_data.csv"
    processor = CryptoDataProcessor(file_name)

    # Preprocess data
    processor.preprocess_data()

    # Display the first few rows
    print(processor.data.head())

    # Get date range
    start_date, end_date = processor.get_date_range()
    print(f"Date range: {start_date} to {end_date}")

    # Display summary statistics
    processor.summary_statistics()

    # Plot technical indicators
    processor.plot_technical_indicators()
    
    # Save the dataset
    output_file_name = "data/processed_crypto_data_with_indicators.csv"
    # processor.save_dataset(output_file_name)
