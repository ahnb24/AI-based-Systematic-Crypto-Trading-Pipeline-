import os
import subprocess
import zipfile
import pandas as pd
import shutil
import logging
import glob
from utils.logging_tools import default_logger
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

class BinanceTradeDownloader:
    def __init__(self, symbol, market_type="spot", start_date="2022-01-01", end_date=None, folder="data", log_file="downloader.log", agg=True):
        self.symbol = symbol.upper()
        self.market_type = market_type
        self.start_date = start_date
        self.end_date = end_date
        # self.folder = os.path.abspath(folder)
        self.folder = folder
        self.agg = agg

        # Set up logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Clean output folder
        # if os.path.exists(self.folder):
        #     self.logger.info(f"Removing existing folder: {self.folder}")
        #     shutil.rmtree(self.folder)
        os.makedirs(self.folder, exist_ok=True)
        self.logger.info(f"Created clean folder: {self.folder}")

    def _ensure_repo(self):
        if not os.path.exists("binance-public-data"):
            self.logger.info("Cloning Binance public data repo...")
            result = subprocess.run(["git", "clone", "https://github.com/binance/binance-public-data.git"], capture_output=True, text=True)
            self.logger.info(result.stdout)
            if result.stderr:
                self.logger.error(result.stderr)
        else:
            self.logger.info("Repo already exists. Skipping clone.")

    def download_data(self):
        self._ensure_repo()
        if self.agg:
            cmd = [
                "python", "binance-public-data/python/download-aggTrade.py",
                "-t", self.market_type,
                "-s", self.symbol,
                "-startDate", self.start_date,
                "-skip-monthly", "1",
            ]
        else:
            cmd = [
                "python", "binance-public-data/python/download-trade.py",
                "-t", self.market_type,
                "-s", self.symbol,
                "-startDate", self.start_date,
                "-skip-monthly", "1",
            ]
        if self.end_date:
            cmd.extend(["-endDate", self.end_date])
        cmd.extend(["-folder", self.folder])

        self.logger.info(f"Running download command: {' '.join(cmd)}")
        
        for attempt in range(10000):
            try:
                # Your code goes here
                print(f"Trying...\nattempt:{attempt}")
                # Example: risky operation
                
                result = subprocess.run(
                cmd,
                input="y\n",  # Automatically answer 'y' to any prompt
                capture_output=True,
                text=True,
                check=True
                )
                self.logger.info("STDOUT:\n" + result.stdout)
                if result.stderr:
                    self.logger.warning("STDERR:\n" + result.stderr)


                break  # Exit the loop if successful
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
        else:
            print("All 10000 attempts failed.")



        # try:
        #     result = subprocess.run(
        #         cmd,
        #         input="y\n",  # Automatically answer 'y' to any prompt
        #         capture_output=True,
        #         text=True,
        #         check=True
        #     )
        #     self.logger.info("STDOUT:\n" + result.stdout)
        #     if result.stderr:
        #         self.logger.warning("STDERR:\n" + result.stderr)
        # except subprocess.CalledProcessError as e:
        #     self.logger.error(f"Subprocess failed with code {e.returncode}")
        #     self.logger.error("STDOUT:\n" + e.stdout)
        #     self.logger.error("STDERR:\n" + e.stderr)




    def extract_and_convert_to_parquet(self, output_file=None):
        if not output_file:
            if self.agg:
                output_file = f"{self.folder}/{self.symbol}_agg_trades.parquet"
            else:
                output_file = f"{self.folder}/{self.symbol}_trades.parquet"

        all_dfs = []

        # Use glob to find all zip files in nested folders
        if self.agg:
            zip_files = glob.glob(os.path.join(self.folder, "data", self.market_type, "daily", "aggTrades", self.symbol, "*", "*.zip"))
        else:
            zip_files = glob.glob(os.path.join(self.folder, "data", self.market_type, "daily", "trades", self.symbol, "*", "*.zip"))
        
        # print(zip_files)
        for zip_path in zip_files:
            # print(zip_path)
            self.logger.info(f"Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.folder)
            # os.remove(zip_path)

        # Now load all CSVs from the folder
        for file in os.listdir(self.folder):
            if file.endswith(".csv"):
                csv_path = os.path.join(self.folder, file)
                self.logger.info(f"Reading CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                if self.agg:
                    df.columns = ['agg_trade_id',         # or 'index' if that's what it represents
                                'price',
                                'qty',
                                'first_trade_id',
                                'last_trade_id',  # appears to be in milliseconds
                                'time',
                                'was_buyer_maker',
                                'was_best_match']
                    
                    # min_ts = 0                     # 1970-01-01
                    # max_ts = 2**63 // 10**6 - 1    # max int64 in milliseconds

                    # df = df[(df['time'] >= min_ts) & (df['time'] <= max_ts)]
                    df['_time'] = pd.to_datetime(df['time'], unit='us')

                    df = df[['_time',
                        'agg_trade_id',         # or 'index' if that's what it represents
                        'price',
                        'qty',
                        'first_trade_id',
                        'last_trade_id',  # appears to be in milliseconds
                        'was_buyer_maker',
                        'was_best_match']]
                    all_dfs.append(df)
                    # os.remove(csv_path)
                else:
                    df.columns = ['trade_id',         # or 'index' if that's what it represents
                                'price',
                                'qty',
                                'quote_qty',
                                'time',  # appears to be in milliseconds
                                'is_buyer_maker',
                                'is_best_match']
                    
                    # min_ts = 0                     # 1970-01-01
                    # max_ts = 2**63 // 10**6 - 1    # max int64 in milliseconds

                    # df = df[(df['time'] >= min_ts) & (df['time'] <= max_ts)]
                    df['_time'] = pd.to_datetime(df['time'], unit='us')

                    df = df[['_time',
                        'trade_id',         # or 'index' if that's what it represents
                        'price',
                        'qty',
                        'quote_qty',
                        'is_buyer_maker',
                        'is_best_match']]
                    all_dfs.append(df)
                    # os.remove(csv_path)

        if all_dfs:
            full_df = pd.concat(all_dfs, ignore_index=True)
            os.makedirs(self.folder, exist_ok=True)
            # Use fastparquet engine to avoid PyArrow extension type issues
            try:
                full_df.to_parquet(output_file, index=False, engine='fastparquet')
                self.logger.info(f"Saved parquet file: {output_file} with {len(full_df)} rows.")
            except ImportError:
                # Fallback to PyArrow with error handling
                try:
                    full_df.to_parquet(output_file, index=False, engine='pyarrow')
                    self.logger.info(f"Saved parquet file: {output_file} with {len(full_df)} rows.")
                except Exception as e:
                    # Final fallback: save as CSV if parquet fails
                    csv_file = output_file.replace('.parquet', '.csv')
                    full_df.to_csv(csv_file, index=False)
                    self.logger.warning(f"Parquet save failed ({e}), saved as CSV instead: {csv_file}")
                    self.logger.info(f"Saved CSV file: {csv_file} with {len(full_df)} rows.")
        else:
            self.logger.warning("No CSV files found to convert.")


    def run(self):
        self.logger.info("Starting BinanceTradeDownloader pipeline.")
        
        self.download_data()
        self.extract_and_convert_to_parquet()
        self.logger.info("Pipeline finished.")

def split_into_yearly_ranges(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    ranges = []

    while start < end:
        next_year = start.replace(year=start.year + 1)
        next_year -= timedelta(days=1)
        range_end = min(next_year, end)
        ranges.append([start.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")])
        start = range_end + timedelta(days=1)  # Avoid duplicate dates

    return ranges



def merge_parquet_files(input_dir, output_file, logger=default_logger):
    # Get all .parquet files in the directory
    input_dir = Path(input_dir)
    parquet_files = list(input_dir.rglob("*.parquet"))


    # Read and concatenate all files
    df_list = [pd.read_parquet(file) for file in parquet_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Save to a single parquet file
    try:
        merged_df.to_parquet(output_file, index=False, engine='fastparquet')
        logger.info(f"Merged {len(parquet_files)} files into {output_file}")
    except ImportError:
        # Fallback to PyArrow with error handling
        try:
            merged_df.to_parquet(output_file, index=False, engine='pyarrow')
            logger.info(f"Merged {len(parquet_files)} files into {output_file}")
        except Exception as e:
            # Final fallback: save as CSV if parquet fails
            csv_file = output_file.replace('.parquet', '.csv')
            merged_df.to_csv(csv_file, index=False)
            logger.warning(f"Parquet save failed ({e}), saved as CSV instead: {csv_file}")
            logger.info(f"Merged {len(parquet_files)} files into {csv_file}")


def get_trades_data(feature_config, start_date="2020-01-01", end_date= None, logger=default_logger, agg=True):
    for symbol in list(feature_config.keys()):
        folder = f'{os.getcwd()}/data/trade_data/{symbol}'
        logger.info("* " * 25)
        logger.info(f"downloading trade data for ----> {symbol}")
        logger.info(f"start_date ----> {start_date}")
        # Configure downloader
        if end_date:
            downloader = BinanceTradeDownloader(
                symbol=symbol,
                agg=agg,
                start_date=start_date,
                end_date=end_date,
                folder=folder,
            )
            downloader.run()
        else:
            now = datetime.now().strftime("%Y-%m-%d")
            # date_list = split_into_yearly_ranges(start_date, now)
            # for date_range in date_list:
            #     print(date_range)
            downloader = BinanceTradeDownloader(
                symbol=symbol,
                start_date=start_date,
                end_date=now,
                folder=folder,
                agg=agg
            )
            downloader.run()
        # merge_parquet_files(folder, f'{folder}/{symbol}__{start_date}__{now}.parquet')

        # Run full pipeline: download → extract → convert to parquet
        
        logger.info("download successful")

