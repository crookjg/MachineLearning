import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler

def describe(df):
    print(df.describe())

def pairs(df):
    sb.pairplot(df)
    plt.show()

def make_choropleths(df):
    # Show map of happiest countries based on 5 year data
    rank = go.Figure(data=go.Choropleth(
        locations=df['Country'],
        z = df['Happiness_Rank'],
        locationmode = 'country names',
        colorscale = 'Viridis',
        colorbar_title = "Happiness Rank Across the World",
    ))

    # Show map of happiness scores based on 5 year data
    score = go.Figure(data=go.Choropleth(
        locations=df['Country'],
        z = df['Happiness_Score'],
        locationmode = 'country names',
        colorscale = 'reds',
        colorbar_title = "Happiness Score Across the World",
    ))

    rank.show()
    score.show()

def show_heatmap(df):
    # Heatmap showing the correlations between the happiness score and other categories
    drop_rank = df.drop(['Country', 'Happiness_Rank'], axis=1)
    correlation = drop_rank.corr()
    heatmap = go.Heatmap(z=np.array(correlation), x=correlation.columns, y=correlation.columns)
    data_hm = [heatmap]
    plot(data_hm)

def top_countries(df):
    ranking = df.sort_values('Happiness_Rank', ascending=True).head(10)
    world_happy = ranking.filter(['Country','Economy','Family','Health','Freedom','Generosity','Trust', 'Dystopia_Residual'])
    world_happy = world_happy.set_index('Country')
    world_happy.plot.barh(stacked=True)
    plt.show()

def score_v_rank(df):
    # Happiness Score vs. Rank Scatter Plot
    plt.scatter(x=df['Happiness_Score'], y=df['Happiness_Rank'], c='DarkBlue')
    plt.xlabel('Happiness Score')
    plt.ylabel('Happiness Rank')
    plt.show()

def drop_generosity(df):
    drop_rank_country = df.drop(['Country', 'Happiness_Rank'], axis=1)
    drop_generosity = drop_rank_country.drop(['Happiness_Score', 'Generosity'], axis=1)

    lm = LinearRegression()
    lm.fit(drop_generosity, drop_rank_country.Happiness_Score)
    print("Estimated Intercept is ", lm.intercept_)
    print("# of coefficients is ", lm.coef_)
    coef_df = pd.DataFrame(list(zip(drop_generosity.columns, lm.coef_)), columns=['features', 'coefficients'])
    print(coef_df)

    plt.scatter(x=lm.predict(drop_generosity), y=drop_rank_country.Happiness_Score, label="Happiness Score vs. Predicted Happiness Score w/o Generosity")
    plt.ylabel('Happiness Score')
    plt.xlabel('Predicted Happiness Score')
    plt.show()

def drop_trust(df):
    drop_rank_country = df.drop(['Country', 'Happiness_Rank'], axis=1)
    drop_trust = drop_rank_country.drop(['Happiness_Score', 'Trust'], axis=1)

    lm = LinearRegression()
    lm.fit(drop_trust, drop_rank_country.Happiness_Score)
    print("Estimated Intercept is ", lm.intercept_)
    print("# of coefficients is ", lm.coef_)
    coef_df = pd.DataFrame(list(zip(drop_trust.columns, lm.coef_)), columns=['features', 'coefficients'])
    print(coef_df)

    plt.scatter(lm.predict(drop_trust), y=drop_rank_country.Happiness_Score, label="Happiness Score vs. Predicted Happiness Score w/o Trust")
    plt.ylabel("Happiness Score")
    plt.xlabel("Predicted Happiness Score")
    plt.show()

def drop_freedom(df):
    drop_rank_country = df.drop(['Country', 'Happiness_Rank'], axis=1)
    drop_freedom = drop_rank_country.drop(['Happiness_Score', 'Freedom'], axis=1)

    lm = LinearRegression()
    lm.fit(drop_freedom, drop_rank_country.Happiness_Score)
    print("Estimated Intercept is ", lm.intercept_)
    print("# of coefficients is ", lm.coef_)
    coef_df = pd.DataFrame(list(zip(drop_freedom.columns, lm.coef_)), columns=['features', 'coefficients'])
    print(coef_df)

    plt.scatter(lm.predict(drop_freedom), y=drop_rank_country.Happiness_Score, label="Happiness Score vs. Predicted Happiness Score w/o Freedom")
    plt.ylabel("Happiness Score")
    plt.xlabel("Predicted Happiness Score")
    plt.show()

def drop_three(df):
    drop_rank_country = df.drop(['Country', 'Happiness_Rank'], axis=1)
    drop_three = drop_rank_country.drop(['Happiness_Score', 'Freedom', 'Trust', 'Generosity'], axis=1)

    lm = LinearRegression()
    lm.fit(drop_three, drop_rank_country.Happiness_Score)
    print("Estimated Intercept is ", lm.intercept_)
    print("# of coefficients is ", lm.coef_)
    coef_df = pd.DataFrame(list(zip(drop_three.columns, lm.coef_)), columns=['features', 'coefficients'])
    print(coef_df)

    plt.scatter(lm.predict(drop_three), y=drop_rank_country.Happiness_Score, label="Happiness Score vs. Predicted Happiness Score w/o Generosity, Trust, Freedom")
    plt.ylabel("Happiness Score")
    plt.xlabel("Predicted Happiness Score")
    plt.show()

def predict_happiness(df):
    # Countries Happiness rank will change each year, so drop this from data frames.
    drop_rank_country = df.drop(['Country', 'Happiness_Rank'], axis=1)
    # Drop the score from the happiness so we can properly predict it.
    drop_happy = drop_rank_country.drop('Happiness_Score', axis=1)

    lm = LinearRegression()
    lm.fit(drop_happy, drop_rank_country.Happiness_Score)
    print("Estimated Intercept is", lm.intercept_)
    print("The number of coefficients in this model are", lm.coef_)

    coef_df = pd.DataFrame(list(zip(drop_happy.columns, lm.coef_)), columns=['features', 'coefficients'])
    print(coef_df)

    plt.scatter(x=lm.predict(drop_happy), y=drop_rank_country.Happiness_Score, label='Happiness Score vs. Predicted Happiness Score')
    plt.ylabel('Happiness Score')
    plt.xlabel('Predicted Happiness Score')
    plt.show()

# 2015
h_15 = pd.read_csv("./MachineLearning/gradProject/Happiness/2015.csv")
h_15.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']
# 2016
h_16 = pd.read_csv("./MachineLearning/gradProject/Happiness/2016.csv")
h_16.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']
# 2017
h_17 = pd.read_csv("./MachineLearning/gradProject/Happiness/2017.csv")
h_17.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']
#2018
h_18 = pd.read_csv("./MachineLearning/gradProject/Happiness/2018.csv")
h_18.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']
# 2019
h_19 = pd.read_csv("./MachineLearning/gradProject/Happiness/2019.csv")
h_19.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']
# 2019
h_20 = pd.read_csv("./MachineLearning/gradProject/Happiness/2020.csv")
h_20.columns = ['Country', 'Happiness_Rank', 'Happiness_Score', 'Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia_Residual']

df = [h_20, h_19, h_18, h_17, h_16, h_15]
happiness = pd.concat(df)

#describe(happiness)
#top_countries(h_20)
#make_choropleths(happiness)
#pairs(happiness)
#show_heatmap(happiness)
#score_v_rank(happiness)
predict_happiness(happiness)
#drop_generosity(happiness)
#drop_trust(happiness)
#drop_freedom(happiness)
#drop_three(happiness)
