def year(growth, rate):
    return math.log(growth) / math.log(1+rate)

def rate(growth, year):
    return pow(growth, 1 / year) - 1

def grow(rate, year):
    return (1 + rate) ** year   
