from functools import cached_property
from trade_sim_util import FMPPriceLoader

class BaseFeature:
    def __init__(self, symbol):
        self.symbol = symbol
        self.value = None
        self.price_loader = FMPPriceLoader()
    
    def value(self):
        if self.value is None:
            self.value = self.calculate()
        return self.value

    def calculate(self):
        raise NotImplementedError

    def get_close_price_during(self, start_date, end_date):
        return self.price_loader.get_close_price_during(self.symbol, start_date, end_date)

    def get_close_price_for_the_last_days(self, last_date, num_days):
        return self.price_loader.get_close_price_for_the_last_days(self.symbol, last_date, num_days)