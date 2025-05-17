# Machine-Learning-Stock
Machine Learning Stock
Stock Price Prediction with Machine Learning

Overview

This repository implements an end-to-end pipeline for time-series forecasting of cryptocurrency and stock prices using deep learning models. It includes data acquisition, cleaning, feature engineering, model training, evaluation, and inference components—primarily built in Jupyter notebooks and modular Python scripts.

Prerequisites

Python 3.8 or higher

pip (Python package installer)

Bybit API credentials (for cryptocurrency data)

CSV data files (optional, notebooks can regenerate via API)

Installation

Clone the repository

git clone https://github.com/idanjo1789/idan.git
cd idan

Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

Install dependencies

pip install -r requirements.txt

Set up API keys
Create a .env file in the project root containing:

BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=1            # set to 0 for mainnet

Project Structure

├── data/                          # Local CSV files and processed datasets

├── models/                        # Saved model weights (e.g., model_BTC.pth)

├── notebooks/                     # Jupyter notebooks for each pipeline step

│   ├── 1_Update_data.ipynb        # Download & combine candlestick data via Bybit API

│   ├── 2_Deletes_incompatible_files.ipynb # Remove unsupported symbols or bad files

│   ├── 3_Constructs_data_testing.ipynb    # Split data & prepare sliding-window indices

│   ├── 4_Building_samples.ipynb           # Generate fixed-length input-output samples

│   ├── 5_Fertile_transformations.ipynb    # Apply feature engineering (FFT, correlations)

│   ├── 6_Builds_trains_model.ipynb        # Define and train ConvTCNModel

│   ├── 7_Checking_the_model.ipynb         # Evaluate performance and visualize results

├── src/                            # Python modules

│   ├── data_utils.py              # Data loading and preprocessing functions

│   ├── model.py                   # ConvTCNModel and training loops

│   └── train.py                   # Script wrapper for model training

├── .env                           # Environment variables (ignored by Git)

├── requirements.txt               # pip dependencies

└── README.md                      # Project overview (this file)


Workflow

Data Acquisition: Run notebooks/1_Update_data.ipynb to fetch 15m candlestick data for selected symbols. Outputs CSV files in data/.

Cleanup: Execute notebooks/2_Deletes_incompatible_files.ipynb to filter out symbols with insufficient history or API errors.

Data Splitting: Use notebooks/3_Constructs_data_testing.ipynb to create train/validation/test splits and index mappings.

Sample Generation: Launch notebooks/4_Building_samples.ipynb to build multi-feature sliding-window samples for model input.

Feature Engineering: Run notebooks/5_Fertile_transformations.ipynb to compute additional features like Fourier transforms and rolling correlations.

Model Training: Open notebooks/6_Builds_trains_model.ipynb or run src/train.py to train the ConvTCNModel on prepared samples, with support for mixed-precision and gradient accumulation.

Evaluation: In notebooks/7_Checking_the_model.ipynb, compute test-set metrics (MSE) and visualize predictions vs. ground truth.

Inference: Load saved weights from models/ and leverage data utilities to forecast new data samples.

Configuration

Hyperparameters and model architecture can be adjusted in notebooks/6_Builds_trains_model.ipynb or via command-line arguments in src/train.py.

Environment variables are loaded using python-dotenv.

Usage

# Run full training from the command line:
python src/train.py \
  --data_dir data/ \
  --symbols BTCUSDT,ETHUSDT \
  --sample_len 2500 \
  --future_steps 100 \
  --batch_size 32 \
  --epochs 20 \
  --use_amp

Contributing

Fork the repository

Create a feature branch: git checkout -b feature/your-feature

Commit changes: git commit -m "Add feature XYZ"

Push: git push origin feature/your-feature

Open a Pull Request

License

This project is licensed under the MIT License.

Contact

Idan josifov – idanjo1789@gmail.com

