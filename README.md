# README: **Deep Reinforcement Learning for Automated Crypto Trading**

## Overview

This repository implements a **Deep Reinforcement Learning (DRL)-based ensemble strategy** for cryptocurrency trading inspired by the research paper:  
**"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"**.  

The core idea revolves around building an **AI Agent** that autonomously learns optimal trading policies through DRL models like PPO, A2C, and DDPG. These agents interact with the cryptocurrency market to maximize returns while adapting to changing market conditions, leveraging dynamic rebalancing and turbulence-based thresholds.

---

## Key Features

- **DRL-based Trading Agents**: Implements popular DRL models such as:
  - **PPO (Proximal Policy Optimization)**
  - **A2C (Advantage Actor-Critic)**
  - **DDPG (Deep Deterministic Policy Gradient)**

- **Ensemble Strategy**: 
  - Combines multiple DRL agents trained on the same data to optimize trading decisions.
  - Uses turbulence metrics to identify market conditions and adjust trading strategies dynamically.

- **Custom Crypto Environment**: Two tailored OpenAI Gym environments:
  - `CryptoEnvTrain`: For training the models on historical data.
  - `CryptoEnvValidation`: For validating the models during rebalancing.

- **Rebalancing Logic**: Implements a periodic rebalancing strategy with validation windows to simulate real-world portfolio adjustments.

- **Performance Metrics**: Tracks Sharpe ratios and other metrics to evaluate agent performance.

---

## AI Agent Concept for Crypto Trading

The AI Agent in this repository leverages **Deep Reinforcement Learning** to autonomously learn trading strategies by interacting with the crypto market environment. Key ideas include:

1. **Market Adaptability**:
   - The agent dynamically adapts to changing market conditions by analyzing turbulence metrics and historical trends.

2. **Self-Learning**:
   - Through DRL algorithms, the agent learns to maximize rewards by optimizing portfolio allocation and reducing drawdowns.

3. **Data-Driven Decision Making**:
   - The agent uses historical crypto price data and technical indicators to simulate trading conditions and improve its policy iteratively.

4. **Autonomous Rebalancing**:
   - The ensemble strategy ensures the portfolio is periodically rebalanced, combining the strengths of multiple models to mitigate risks.

---

## Repository Structure

```
├── data/                          # Directory for storing datasets
│   └── processed_crypto_data_with_indicators.csv  # Example processed dataset with indicators
├── env/                           # Custom environments for DRL
│   ├── EnvMultipleCrypto_train.py         # Training environment
│   └── EnvMultipleCrypto_validation.py    # Validation environment
├── scripts/
│   ├── preprocess_data.py         # Data preprocessing script
│   ├── data_downloader.py         # Script to fetch raw cryptocurrency data
│   └── model_runner.py            # Core script for training and running the ensemble strategy
├── README.md                      # Project documentation
└── trained_models/                # Directory for saving trained DRL models
```

---

## Requirements

Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

### Key Libraries
- **Stable Baselines3**: DRL algorithms like PPO, A2C, and DDPG.
- **Stockstats**: For calculating technical indicators.
- **Pandas and Numpy**: For data manipulation and analysis.
- **Matplotlib**: For visualizations.

---

## Getting Started

### 1. Preprocess the Data
Ensure you have cryptocurrency price data. Use `preprocess_data.py` to clean and generate necessary technical indicators.

```bash
python preprocess_data.py --input raw_data.csv --output processed_crypto_data_with_indicators.csv
```

### 2. Train the AI Agents
Run `model_runner.py` to train DRL agents and execute the ensemble strategy.

```bash
python model_runner.py
```

The script will:
- Load the preprocessed dataset.
- Train the DRL models (PPO, A2C, DDPG).
- Rebalance the portfolio periodically.

### 3. Analyze Results
- Trained models are saved in the `trained_models/` directory.
- Performance metrics and logs are printed during execution.

---

## Example Workflow

1. **Prepare Dataset**: 
   - Download historical cryptocurrency data using `data_downloader.py`.
   - Process it into a format compatible with the DRL environment.

2. **Run DRL Agents**:
   - Train the agents on historical data using the custom training environment.
   - Validate and rebalance portfolios periodically.

3. **Evaluate Performance**:
   - Use metrics like Sharpe ratio to evaluate the ensemble strategy's effectiveness.

---

## Future Improvements

- **Live Trading Integration**: Extend the AI agents for real-time trading using APIs like Binance or Coinbase.
- **Advanced Models**: Experiment with more advanced DRL algorithms like SAC or TD3.
- **Extended Features**: Incorporate more sophisticated indicators, sentiment analysis, or macroeconomic factors.

---

## References

- **Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy**  
  *Jiang, Zhu, and Liang (2020)*

---

## Contribution

Feel free to open issues or contribute via pull requests to improve the repository. Suggestions for new features or optimizations are welcome!

---

## License

This project is open-source and licensed under the [MIT License](LICENSE).

---

## Contact

For any inquiries, please open an issue or reach out via email.
