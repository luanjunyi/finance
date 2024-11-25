from functools import cached_property
from fmp_data import FMPPriceLoader

class BaseFeature:
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_loader = FMPPriceLoader()
    
    @cached_property
    def value(self):
        return self.calculate()

    def calculate(self):
        raise NotImplementedError

    def get_close_price_during(self, start_date, end_date):
        return self.price_loader.get_close_price_during(self.symbol, start_date, end_date)

    def get_close_price_for_the_last_days(self, last_date, num_days):
        return self.price_loader.get_close_price_for_the_last_days(self.symbol, last_date, num_days)