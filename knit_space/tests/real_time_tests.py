import random
import uuid
import re
import csv # For reading the stock ticker list
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional
import logging

from .base  import AbstractQATest, QAItem, create_test_cases, register_test


try:
    import yfinance as yf
except ImportError:
    print("WARNING: yfinance library is not installed. LiveStockPriceQATest may not function fully.")
    pass

@register_test('live_knowledge', 'stock_price', 'finance', 'external_data')
class LiveStockPriceQATest(AbstractQATest):
    DEFAULT_TICKER_CSV_PATH = "data/stock_tickers.csv" # Adjust as needed
    DEFAULT_VERIFICATION_TOLERANCE = 0.02 # 2% tolerance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ticker_csv_path = self.config.get("ticker_csv_path", self.DEFAULT_TICKER_CSV_PATH)
        self.verification_tolerance = self.config.get(
            "stock_price_tolerance", self.DEFAULT_VERIFICATION_TOLERANCE
        )
        self.stock_list: List[Dict[str, str]] = []
        self._load_stock_list()

        if 'yf' not in globals():
            self.logger.error("yfinance library is not imported. Stock price fetching will fail.")
            self._yfinance_available = False
        else:
            self._yfinance_available = True


    def _load_stock_list(self):
        try:
            path_to_load = self.ticker_csv_path
            if not os.path.isabs(path_to_load):
                 # Assuming this test file is in some directory, and 'data/' is sibling to it or at project root
                 # This might need adjustment based on your project structure.
                 base_dir = os.path.dirname(os.path.abspath(__file__))
                 path_to_load = os.path.join(base_dir, self.ticker_csv_path)
                 # A common pattern is to have a project root and data is relative to that.
                 # If that's the case, config might pass an absolute path or a path resolver.

            if not os.path.exists(path_to_load):
                self.logger.warning(f"Ticker CSV file not found at resolved path: {path_to_load}. Trying original path: {self.ticker_csv_path}")
                path_to_load = self.ticker_csv_path # Try original path as is

            if not os.path.exists(path_to_load):
                 self.logger.error(f"Stock ticker CSV file not found at {path_to_load}. Test may not generate many diverse items.")
                 # Add a few common fallbacks if file not found
                 self.stock_list = [
                    {"Ticker": "AAPL", "Name": "Apple Inc."},
                    {"Ticker": "MSFT", "Name": "Microsoft Corp."},
                    {"Ticker": "GOOGL", "Name": "Alphabet Inc."},
                    {"Ticker": "AMZN", "Name": "Amazon.com, Inc."},
                    {"Ticker": "NVDA", "Name": "NVIDIA Corporation"},
                    {"Ticker": "META", "Name": "Meta Platforms, Inc."},
                    {"Ticker": "TSLA", "Name": "Tesla, Inc."},
                    {"Ticker": "NFLX", "Name": "Netflix, Inc."},
                    {"Ticker": "ADBE", "Name": "Adobe Inc."},
                    {"Ticker": "CRM", "Name": "Salesforce, Inc."},
                    {"Ticker": "INTC", "Name": "Intel Corporation"},
                    {"Ticker": "CSCO", "Name": "Cisco Systems, Inc."}
                 ]
                 return

            with open(path_to_load, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                # Assuming CSV has 'Ticker' and 'Name' columns. Adjust if different.
                for row in reader:
                    if row.get('Ticker') and row.get('Name'):
                        self.stock_list.append({'Ticker': row['Ticker'].strip(), 'Name': row['Name'].strip()})
            if not self.stock_list:
                 self.logger.error(f"No stocks loaded from {path_to_load}. Check CSV format (expected Ticker, Name columns).")
            else:
                 self.logger.info(f"Loaded {len(self.stock_list)} stocks from {path_to_load}.")
        except Exception as e:
            self.logger.error(f"Error loading stock list from {self.ticker_csv_path}: {e}", exc_info=True)
            # Add fallbacks if loading fails
            if not self.stock_list:
                self.stock_list = [
                    {"Ticker": "TSLA", "Name": "Tesla Inc."}, {"Ticker": "NVDA", "Name": "NVIDIA Corporation"}
                ]


    def _get_previous_close_price(self, ticker_symbol: str) -> Optional[Dict[str, Any]]:
        if not self._yfinance_available:
            self.logger.error("yfinance not available, cannot fetch price.")
            return None
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info # Use .info for more comprehensive data, including 'previousClose'
            
            prev_close = info.get('previousClose')
            currency = info.get('currency', 'USD')

            if prev_close is None:
                self.logger.warning(f"'previousClose' not found directly in .info for {ticker_symbol}. This can happen for some tickers/exchanges.")
                # As a simple fallback, if not found, we can't proceed for this specific test target ("previous close")
                return None
            
            return {
                "price": float(prev_close),
                "currency": currency,
                "price_type": "previous_close",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            # yfinance can sometimes raise errors for delisted/problematic tickers, or network issues
            self.logger.error(f"Failed to fetch stock info for {ticker_symbol} via yfinance: {e}")
            return None

    def generate(self, count: int = 3, **kwargs) -> Iterator[QAItem]:
        if not self._yfinance_available:
            self.logger.error(f"{self.name}: yfinance not available. Skipping generation.")
            return
        if not self.stock_list:
            self.logger.error(f"{self.name}: Stock list is empty. Cannot generate items. Check ticker CSV path/format.")
            return

        for _ in range(count):
            selected_stock_info = random.choice(self.stock_list)
            ticker = selected_stock_info['Ticker']
            company_name = selected_stock_info['Name']

            price_data = self._get_previous_close_price(ticker)

            if not price_data:
                self.logger.warning(f"Skipping QAItem for {ticker} ({company_name}) due to price fetch failure.")
                continue # Try next item if in a loop, or just return if count was 1

            actual_price = price_data['price']
            price_currency = price_data['currency']

            question_text = (
                f"What was the previous day's closing stock price for {company_name} ({ticker})? "
                f"Provide your answer as a numerical value within <answer></answer> tags. "
                f"For example: <answer>175.32</answer>"
            )
            item_id = f"{self.name}-{ticker}-{uuid.uuid4().hex[:6]}"

            yield QAItem(
                id=item_id,
                question=question_text,
                answer=actual_price, # The numerical previous close price
                skill_coefficient = 2,
                modality='text',
                metadata={
                    'ticker': ticker,
                    'company_name': company_name,
                    'retrieved_price_timestamp': price_data['timestamp'],
                    'retrieved_price_type': price_data['price_type'],
                    'price_currency': price_currency,
                    'verification_tolerance_percentage': self.verification_tolerance,
                    'data_source_for_verification': "Yahoo Finance via yfinance",
                    'output_format_instruction': "<answer>PRICE</answer>"
                },
                verification_fn=self._verify_stock_price_answer
            )

    @staticmethod
    def _verify_stock_price_answer(expected_price: float, provided_answer_str: str, qa_item: QAItem) -> bool:
        logger = logging.getLogger("LiveStockPriceQATest.VerificationFn") # Generic
        if hasattr(qa_item, 'logger') and qa_item.logger: logger = qa_item.logger

        match = re.fullmatch(r'<answer>([0-9.]+)</answer>', provided_answer_str.strip(), re.IGNORECASE)
        if not match:
            logger.warning(f"VFY {qa_item.id}: Format mismatch. Expected '<answer>PRICE</answer>', Got: '{provided_answer_str}'")
            return False

        try:
            llm_price_str = match.group(1)
            # Handle potential commas if LLM uses them as thousands separators, though question asks for numerical value
            llm_price_str = llm_price_str.replace(',', '')
            llm_price = float(llm_price_str)
        except ValueError:
            logger.warning(f"VFY {qa_item.id}: Non-float value in answer tag. Got: '{match.group(1)}'")
            return False

        tolerance_percent = qa_item.metadata.get('verification_tolerance_percentage', 0.02) # Default to 2% if not in meta
        
        # Calculate the allowed absolute difference
        allowed_difference = abs(expected_price * tolerance_percent)
        
        is_correct = abs(llm_price - expected_price) <= allowed_difference
        
        log_level = logging.INFO if is_correct else logging.WARNING
        logger.log(log_level,
                   f"Stock Price VFY {('PASSED' if is_correct else 'FAILED')} for {qa_item.id} ({qa_item.metadata.get('ticker')}): "
                   f"Exp:'{expected_price:.2f}', LLM:'{llm_price:.2f}', "
                   f"Diff:'{abs(llm_price - expected_price):.2f}', AllowedDiff:'{allowed_difference:.2f}' ({tolerance_percent*100:.1f}%).")
        return is_correct