

import pandas as pd 

from datetime import datetime,timedelta

from matplotlib import pyplot as plt

from matplotlib import dates as mpl_dates

import matplotlib.pyplot as plt
# # print(plt.style.availabl

jls_extract_var = plt
jls_extract_var.style.use('seaborn')


dates = [

    datetime(2021, 5, 24),
    datetime(2021, 5, 25),

    datetime(2021, 5, 26),

    datetime(2021, 5, 27),

    datetime(2021, 5, 28),

    datetime(2021, 5, 29),

    datetime(2021, 5, 30),

]

y=[0,1,3,4,6,5,7]

plt.plot_date(dates,y,linestyle='solid')

plt.tight_layout()


