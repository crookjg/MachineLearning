import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler
import seaborn as sb

"""
ax = fig.add_subplot(321)
countries = happiness['Country'].head(10)
economy = happiness['Economy'].head(10)
family = happiness['Family'].head(10)
health = happiness['Health'].head(10)
freedom = happiness['Freedom'].head(10)
trust = happiness['Trust'].head(10)
gen = happiness['Generosity'].head(10)
dys = happiness['Dystopia_Residual'].head(10)

ax.bar(countries, economy, label='Economy')
ax.bar(countries, family, bottom=economy, label='Family')
ax.bar(countries, health, bottom=family+economy, label='Health')
ax.bar(countries, freedom, bottom=health+family+economy, label='Freedom')
ax.bar(countries, trust, bottom=freedom+health+family+economy, label='Trust')
ax.bar(countries, gen, bottom=trust+freedom+health+family+economy, label='Generosity')
ax.bar(countries, dys, bottom=gen+trust+freedom+health+family+economy, label='Dystopia Residual')
plt.legend()
plt.xlabel('Countries')
plt.ylabel('Happiness Score')
"""

plt.show()
