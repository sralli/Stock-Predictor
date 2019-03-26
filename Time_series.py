import pandas as pd 
import pandas_datareader as pdr
import datetime
from dateutil.relativedelta import relativedelta
import fix_yahoo_finance as fix 

class Stock_Time_Series():
    def __init__(self, to_date = datetime.datetime.now(), year_span = 5):
        #to_date = current date
        # Amount of years to use = 5;

        start_date = to_date - relativedelta(years = year_span)
        self.dates = pd.date_range(start_date, to_date)

    def fetch_data(self, symbol_list):
        #Returns pands df of each symbol:

        good_sym = []
        bad_sym = []

        if 'SPY' not in symbol_list:
            symbol_list.insert(0, 'SPY')

        #Creating base df from SPY
        try: 
            df = self.__fetch_data('SPY')
            good_sym.append('SPY')
        except: 
            bad_sym.append('SPY')

        #Parse additional symbols and add it:

        for sym in symbol_list:
            try: 
                if sym=='SPY':
                    continue
                
                data_df = self.__fetch_data(sym)
                df[sym] = data_df
                good_sym.append(sym)
            
            except Exception as e:
                bad_sym.append(sym)
        
        df = df.dropna(subset=['SPY'])
        df.fillna(method='ffill', inplace = True)
        df.fillna(method='bfill', inplace = True)


        return df, good_sym, bad_sym

    def __fetch_data(self, sym):
        #Returns pd df from yahoo:
         #pdr.get_data_yahoo(self.ticker, self.start, self.end)
        fetched = pdr.get_data_yahoo(sym, self.dates[0], self.dates[-1])
        return pd.DataFrame(data = fetched, columns=[sym], index = fetched.index.values)