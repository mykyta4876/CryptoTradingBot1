import ta
import pandas as pd 

def buy_condition(row): #entry
        if row['MA-st'] > row['MA-lt']:
            return True
        else:
            return False
        
    #     return row['MA-st'] > row['MA-lt']:

def sell_condition(row): #exit
    if row['MA-st'] < row['MA-lt']:
        return True
    else:
        return False
    

def populate_indicators(data:pd.DataFrame):
        data['MA-st'] = ta.trend.sma_indicator(data['close'], 10)
        data['MA-lt'] = ta.trend.sma_indicator(data['close'], 40)
        data['EMAf'] = ta.trend.ema_indicator(data['close'], 10)
        data['EMAs'] = ta.trend.ema_indicator(data['close'], 30)
        data['Trend'] = ta.trend.sma_indicator(data['close'], 50)
        data['RSI'] = ta.momentum.rsi(data['close'])
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

        # MACD = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
        # data['MACD'] = MACD.macd()
        # data['MACD_histo'] = MACD.macd_diff()
        # data['MACD_signal'] = MACD.macd_signal()

        # BB = ta.volatility.BollingerBands(close=data['close'], window=100, window_dev=2)
        # data["BB_lower"] = BB.bollinger_lband()
        # data["BB_upper"] = BB.bollinger_hband()
        # data["BB_avg"] = BB.bollinger_mavg()
        return data
    

class Strategy:
    def __init__(self, data, strategy='s1'):
        self.data = data
        self.strategy = strategy
        
        self.data = populate_indicators(self.data)
        self.previous_row = self.data.iloc[0]
        
        self.data['short_entry'] = ''
        self.data['short_exit'] = ''
        self.data['long_entry'] = ''
        self.data['long_exit'] = ''
        
        self.run()
    
    
    
    def check_short_entry_condition(self, row, previous_row):
        if self.strategy == 's1':
            return row['close'] < row['Trend'] and row['EMAf'] < row['EMAs'] and previous_row['EMAf'] > previous_row['EMAs'] and row['RSI'] > 30
        else:
            pass


    def check_short_exit_condition(self, row, previous_row):
        if self.strategy == 's1': 
            return row['EMAf'] > row['EMAs'] and previous_row['EMAf'] < previous_row['EMAs']
        else:
            pass


    def compute_short_sl_level(self, row, entry_price):
        if self.strategy == 's1': 
            return entry_price + 2 * row['ATR']
        else:
            pass


    def compute_short_tp_level(self, row, entry_price, stop_loss_price):
        risk_reward_ratio = 4
        if self.strategy == 's1':
            return entry_price * (1 - risk_reward_ratio * (stop_loss_price / entry_price - 1))
        else:
            pass

    def check_long_entry_condition(self, row, previous_row):
        if self.strategy == 's1':
            return row['close'] > row['Trend'] and row['EMAf'] > row['EMAs'] and previous_row['EMAf'] < previous_row['EMAs'] and row['RSI'] < 70
        else:
            pass


    def check_long_exit_condition(self, row, previous_row):
        if self.strategy == 's1':
            return row['EMAf'] < row['EMAs'] and previous_row['EMAf'] > previous_row['EMAs']
        else:
            pass


    def compute_long_sl_level(self, row, entry_price):
        return entry_price - 2 * row['ATR']


    def compute_long_tp_level(row, entry_price, stop_loss_price):
        risk_reward_ratio = 4
        return entry_price * (1 + risk_reward_ratio * (1 - stop_loss_price / entry_price))
    
    def run(self):
        
        for index, row in self.data.iterrows():
            price = row['close']
            self.data.at[index, 'long_exit'] = self.check_long_exit_condition(row, self.previous_row)    
            self.data.at[index, 'short_exit'] = self.check_short_exit_condition(row, self.previous_row)
            self.data.at[index, 'long_entry'] = self.check_long_entry_condition(row, self.previous_row)
            self.data.at[index, 'short_entry'] = self.check_short_entry_condition(row, self.previous_row)
            self.previous_row = row
        # return self.data
        # pass