def next_quarter(period: str):
 
    year, quarter = period.split()
    year = int(year)
    q = int(quarter.replace("Q", ""))
 
    if q == 4:
        return f"{year+1} Q1"
    else:
        return f"{year} Q{q+1}"
 