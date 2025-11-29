import torch
import matplotlib.pyplot as plt
from config import get_config
from data import make_env

def test_nyse_dataset():
    # 1. Load Config
    cfg = get_config("generic")
    cfg.ENV.ENV_NAME = "finance"
    
    # 2. Create Environment (Triggers download)
    env = make_env(cfg)
    
    # 3. Visualization
    print("\nVisualizing NYSE(N) Market Dynamics...")
    full_traj = env.get_full_trajectory().numpy()
    
    # Extract just the stock data (exclude time index)
    stocks_data = full_traj[:, :-1] 
    
    # Cumulative Return (Wealth Growth) for visualization
    # We take cumulative product of price relatives to reconstruct price curves
    cumulative_wealth = torch.tensor(stocks_data).cumprod(dim=0)
    
    plt.figure(figsize=(12, 6))
    # Plot first 5 stocks as samples
    for i in range(min(5, env.num_assets)):
        ticker_name = env.clean_df.columns[i]
        plt.plot(cumulative_wealth[:, i], label=ticker_name, linewidth=1)
        
    plt.title(f"NYSE(N) Dataset Reconstruction (1985-2010)\n{env.num_assets} Stocks, {env.num_timesteps} Days")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Wealth ($1 invested)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("nyse_data_check.png")
    print("Saved plot to 'nyse_data_check.png'")
    
    # 4. Sanity Check for Koopman
    print(f"\nData Stats:")
    print(f"Min Relative: {stocks_data.min():.4f}")
    print(f"Max Relative: {stocks_data.max():.4f}")
    print(f"Mean Relative: {stocks_data.mean():.4f} (Should be > 1.0 for growing market)")

if __name__ == "__main__":
    test_nyse_dataset()