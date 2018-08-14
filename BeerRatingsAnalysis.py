
# coding: utf-8

# # Project: Multiple Regression Analysis for Beer Ratings

# ### Kate Grosch and Lucas Baker

# We would like to determine whether factors such as ABV, reviewer age, and beer appearance contribute to the overall rating of a beer, using data from online craft beer ratings.
# 
# To begin our analysis, we will load the project packages and the dataset.

# In[1]:


# loading the packages and modules

get_ipython().run_line_magic('matplotlib', 'inline')

# general packages
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from collections import defaultdict

# for statistics
import scipy.stats as stats
import statsmodels.api as sm

# for visualizations
import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")


# ### Important Notes about the Data
# 
# We do some basic data cleaning below: We remove ABV values outside of 3 standard deviations from the mean (to remove some junk ABV values we had) and we remove rows without a valid reviewer gender and age. 

# In[2]:


#loading and cleaning the data

df = pd.read_csv("beer.csv")

df = df.rename(index=str, columns={'beer/ABV': 'ABV', 
                                   'user/gender':'gender', 
                                   'user/ageInSeconds':'reviewerAgeInSeconds', 
                                   'review/overall':'overall', 
                                   'beer/beerId':'beerId', 
                                   'beer/brewerId':'brewerId', 
                                   'review/appearance':'appearance', 
                                   'review/aroma':'aroma', 
                                   'review/palate':'palate', 
                                   'review/taste':'taste', 
                                   'review/timeUnix': 'unixPostTime'})

# removing outliers and null values
df = df[np.abs(df.ABV-df.ABV.mean()) <= (3*df.ABV.std())]
df = df[df.reviewerAgeInSeconds < 2838240000]
df = df[df.reviewerAgeInSeconds.notnull()]
df = df[df.gender.notnull()]

Y = df['overall']
df = df.drop(['review/text', 
              'review/timeStruct',
              'user/birthdayRaw',
              'user/birthdayUnix',
              'index',
              'beer/style', 
              'user/profileName'], axis=1)

df.describe()


# ### (1) Model and Variables
# 
# Our full list of variables is ABV, brewer, appearance rating, aroma rating, palate rating, taste rating, review submission date, reviewer age, and reviewer gender. We got this dataset from Kaggle, and the CSV is available at https://github.com/katiegrosch/beer-data-analysis.
# 
# Before performing the analysis, we will preprocess the data to extract our explanatory features. Our desired y and x<sub>i</sub>'s are as follows:
# 
# y: Overall rating, on a range of [1.0, 5.0]. <br>
# x<sub>1</sub> The number of reviews for the beer. <br>
# x<sub>2</sub> Average reviewer age. <br>
# x<sub>3</sub> Fraction of female reviewers. <br>
# x<sub>4</sub> Beer appearance rating, in the range [1.0, 5.0]. <br>
# x<sub>5</sub> Alcohol by volume. <br>
# x<sub>6</sub> Length of beer name. <br>
# 
# We will be modeling the impact of the x<sub>i</sub>'s on y in the form of the following multiple regression model, where &beta;<sub>i</sub>'s are constants fit by least squares regression:<br><br>
# y = &beta;<sub>0</sub> + &beta;<sub>1</sub>x<sub>1</sub> + &beta;<sub>2</sub>x<sub>2</sub> + &beta;<sub>3</sub>x<sub>3</sub> + &beta;<sub>4</sub>x<sub>4</sub> + &beta;<sub>5</sub>x<sub>5</sub> + &beta;<sub>6</sub>x<sub>6</sub>
# 
# Let's reshape and group the data by beer:

# In[3]:


df['beerNameLength'] = pd.Series([len(x) for x in df['beer/name']], index=df.index)

beer_abv = dict()
beer_overall = defaultdict(list)
beer_appearance = defaultdict(list)
reviewer_ages = defaultdict(list)
reviewer_genders = defaultdict(list)
for index, row in df.iterrows():
    name = row['beer/name']
    beer_abv[name] = row['ABV']  # Will overwrite, but same values
    beer_overall[name].append(row['overall'])
    beer_appearance[name].append(row['appearance'])
    reviewer_ages[name].append(row['reviewerAgeInSeconds'])
    reviewer_genders[name].append(0.0 if row['gender'].startswith("M") else 1.0)
names = sorted(set(df['beer/name']))
    
combined = {
    'beerName': names,
    'numReviews': [len(beer_overall[name]) for name in names],
    'ABV': [beer_abv[name] for name in names],
    'avgAppearance': [np.mean(beer_appearance[name]) for name in names],
    'avgReviewerAgeInYears': [np.mean(reviewer_ages[name]) /
                              31557600 for name in names],
    'fractionFemale': [np.mean(reviewer_genders[name]) for name in names],
    'beerNameLength': [len(name) for name in names],
    'avgOverall': [np.mean(beer_overall[name]) for name in names]
}

# Save for later
original = df.drop(['beerId', 
                    'brewerId',
                    'unixPostTime',
                    'reviewerAgeInSeconds',
                    'gender'
                   ], axis=1)
df = pd.DataFrame(combined)
df.describe()


# To begin our analysis, let's look at a graph of every x<sub>i</sub> versus our target variable (overall rating).

# In[4]:


Y = df['avgOverall']
ratings = df.drop(['avgOverall', 'beerName'], axis=1)

for column in ['avgAppearance',
               'numReviews',
               'ABV',
               'beerNameLength',
               'avgReviewerAgeInYears',
               'fractionFemale']:
    plt.figure(figsize=(4, 3))
    plt.scatter(ratings[column], Y)
    plt.ylabel('Overall Rating', size=15)
    plt.xlabel(column, size=15)


# ### Variable Correlations
# 
# Visually speaking, there appears to be a strong correlation between appearance rating and overall rating, as might be expected. Number of reviews also appears at least somewhat linked to quality: while there are many beers with few reviews at all levels, the most frequently reviewed beers are rated above 4.0, relative to a mean average overall rating of 3.67. The other explanatory variables seem to have a more tenuous connection. Plots for ABV, beer name length, and average reviewer age in years appear as a fairly undifferentiated mass, and while it is clear that female reviewers give a higher rating on average, the average fraction of female viewers is 1.1%! Detecting a strong effect on such a lopsided distribution may be difficult.
# 
# Let's also see if any of the variables correlate with each other. If we encounter multicollinearity, it may make sense to remove one of the correlated variables:

# In[5]:


corrmat = ratings.corr()
sns.heatmap(corrmat, vmax=.4);


# ### Variable correlations in the heatmap
# 
# In the confusion matrix above, lighter colors imply higher correlation. Trivially, each variable has a perfect correlation with itself. We can confirm that none of these variables correlate strongly with one another, which is reasonable given the idiosyncratic selection. However, ABV and appearance rating show at least a weak relationship, which may suggest a correlation between ABV, appearance, and a third variable such as style. (For example, an Imperial Stout might tend to be both stronger and darker in color.)
# 
# Before we create our regression model, let's look at a histogram of the ratings to get a sense of where they cluster. It looks like the ratings are left-skewed, but overall have a near-normal distribution. We confirm by calculating the skew of the data to be -0.9899, which might affect our model.

# In[6]:


plt.hist(Y)
plt.xlabel("Overall Rating Score: $Y_i$")
plt.ylabel("Number of Ratings")
plt.title("Histogram of Rating Frequency")

print("Skew: %f" % Y.skew())


# ### (2) MLR Parameter Estimation and Confidence Intervals

# In[7]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr = lr.fit(ratings, Y)
y_pred = lr.predict(ratings)

plt.scatter(Y, y_pred)
plt.xlabel("Ratings: $Y_i$")
plt.ylabel("Predicted ratings: $\hat{Y}_i$")
plt.title("Ratings vs Predicted ratings: $Y_i$ vs $\hat{Y}_i$")


# In[8]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y, y_pred)
print("Mean Squared Error: %f" % mse)


# The variance of the estimator &sigma;<sup>2</sup> is close to MSE, but takes into account the degrees of freedom lost by the 6 regressors and the intercept:

# In[9]:


print("Variance of error: SSE / (n - (k + 1)) = ", mse * len(Y) / (len(Y) - 7))


# In[10]:


print("Beta coefficients: ")
print(lr.coef_)
print("Largest coefficient: %f" % np.amax(lr.coef_))
print("R-squared: %f" % sklearn.metrics.r2_score(Y, y_pred))


# The R-squared value is disappointingly low, but not so low as to suggest that our predictors are useless. Does the model have any significant variables?
# 
# The time has come to calculate confidence intervals, but remarkably, scikit-learn does not provide a built-in method of accessing model statistics. Let's switch to statsmodel, which does. The scikit model will provide a reference by which to confirm our numbers. (This sanity check is quite valuable: for instance, if the intercept is not explicitly added in statsmodel, it will not be included and the resulting R-squared will be unreasonably high.)

# In[11]:


ratings['intercept'] = np.ones(len(ratings))
reg = sm.OLS(Y, ratings).fit()
y_pred = reg.predict(ratings)

plt.scatter(Y, y_pred)
plt.xlabel("Ratings: $Y_i$")
plt.ylabel("Predicted ratings: $\hat{Y}_i$")
plt.title("Ratings vs Predicted ratings: $Y_i$ vs $\hat{Y}_i$")

print(reg.summary())


# As shown, the 95% (&alpha; = .05) confidence intervals for the regressors are as follows:
# 
# CI(&beta;<sub>intercept</sub>, &alpha; = .05) = [0.833, 1.548] <br>
# CI(&beta;<sub>numReviews</sub>, .05) = [-0.001, 0.002] <br>
# CI(&beta;<sub>ABV</sub>, .05) = [-0.018, 0.022] <br>
# CI(&beta;<sub>avgAppearance</sub>, .05) = [0.677, 0.823] <br>
# CI(&beta;<sub>avgReviewerAgeInYears</sub>, .05) = [-0.013, -0.001] <br>
# CI(&beta;<sub>fractionFemale</sub>, .05) = [0.071, 1.001] <br>
# CI(&beta;<sub>beerNameLength</sub>, .05) = [-0.005, 0.003] <br>
# 
# A one-star increase in appearance rating, all else equal, produces an extra .75 stars in overall rating. Women also rate just over half a star higher, and older reviewers about .0074 stars lower per year. All three of these variables are significant at the &alpha; = .05 level, even the reviewer age, which is an excellent demonstration of the distinction between significance and effect size. The other three variables (name length, number of reviews, and ABV) have high p-values, are not significant, and bear little apparent relation to overall rating.
# 
# ### (3, 4) Test for Significance of Regression & Final Model Building
# 
# The F-statistic of the model is 78.09, well over the critical value, as suggested by the F-test p-value of 9.20e-76. Thus, despite the quirky choice of variables, the resulting model is clearly significant.
# 
# Having evaluated the significance of all regressors, we will leave in appearance rating, fraction of female reviewers, and average reviewer age while removing the others to build the final model.

# In[12]:


ratings_final = ratings.drop(['numReviews', 'ABV', 'beerNameLength'], axis=1)
reg_final = sm.OLS(Y, ratings_final).fit()
y_pred_final = reg_final.predict(ratings_final)

plt.scatter(Y, y_pred_final)
plt.xlabel("Ratings: $Y_i$")
plt.ylabel("Predicted ratings: $\hat{Y}_i$")
plt.title("Ratings vs Predicted ratings: $Y_i$ vs $\hat{Y}_i$")

print(reg_final.summary())


# In the new model, the F-statistic has roughly doubled (156.3 vs 78.09) and the p-value dropped by a factor of 1000 (9.24e-79 vs 9.20e-76), while the adjusted R-squared has also increased slightly (0.381 vs 0.379). The raw R-squared has decreased slightly (0.383 vs 0.384), but this is to be expected because additional regressors can only increase the R-squared value. From the rise in adjusted R-squared and F-statistic values, we may conclude that the new model with fewer variables is superior.
# 
# We would also like to look at the importance of each &beta;<sub>i</sub>. How large is the average impact of each variable on the final prediction? This can be evaluated by looking at the product of each &beta;<sub>i</sub> with its average, minimum, and maximum input.

# In[13]:


for c in ['avgAppearance', 'avgReviewerAgeInYears', 'fractionFemale']:
    col = ratings_final[c]
    vals = [min(col), np.mean(col), max(col)]
    print('Min, mean, max effect sizes for %s:' % c)
    print('%f, %f, %f\n' % tuple(reg_final.params[c] * x for x in vals))


# Appearance is obviously the most important predictor, but surprisingly, the impact of reviewer age is much larger than that of gender even though the gender coefficient is much larger. This is an issue of units: if we had used, for instance, fractions of an average human lifespan of 80 years, the reviewer age coefficient would have been much higher in magnitude. In any case, we can conclude that the order of regressor importance is (1) average appearance, (2) average reviewer age, and (3) fraction female. 
# 
# ### (5) Analysis of Residuals
# 
# Looking back, does the error in our data fit the Gaussian assumptions of a normal, zero-mean, constant variance random variable? Earlier, we calculated the skew to be -0.9899, so the ratings themselves are not quite normally distributed. However, multiple linear regression can still work effectively if the residuals are near-random. We now plot the residuals versus the predictions as well as each x<sub>i</sub> to confirm this assumption:

# In[14]:


columns = [y_pred_final]
columns.extend(ratings_final[c] for c in ['avgAppearance',
                                          'avgReviewerAgeInYears',
                                          'fractionFemale'])

for i, name in enumerate(['prediction',
                          'avgAppearance',
                          'avgReviewerAgeInYears',
                          'fractionFemale']):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.scatter(columns[i], Y - y_pred_final)
    plt.ylabel('Residuals', size=15)
    plt.xlabel(name, size=15)
    plt.subplot(1, 2, 2)
    plt.hist(Y - y_pred_final, 10, orientation='horizontal')


# The distribution of the data is not ideal, as we would much prefer to see an even spread of values along the x-axis. However, there is no clear evidence of heteroscedasticity, nonlinear patterns, or serious outliers.
# 
# There does appear to be asymmetry across the x-axis, where there are more negative residuals, with higher average magnitudes, than the positive ones. This imbalance reflects the skew in the initial distribution, where ratings are generally clustered around the 3-4 star range and rarely fall below. Since ratings are bounded at 1.0 and 5.0, there is more room for error by predicting too low than too high. Interpreting this asymmetry is a judgment call, but there is nothing here to suggest that the data would be better served by something other than a linear model.

# ## (6, 7) Discussion

# Our project aimed to investigate how well multiple linear regression can predict overall beer rating based on the six features selected. We report that our selection of variables predicts overall rating with significance, but with low precision. One variable, average appearance rating, correlated extremely well with overall rating, while the fraction of female reviewers and average reviewer age also proved significant at the &alpha; = .05 level. The other three variables, ABV, beer name length, and number of reviews, showed no significant predictive ability. Our final model predicted overall rating from appearance rating, fraction of female reviewers, and average reviewer age with an R-squared of 0.383 and adjusted R-squared of 0.381, an F-statistic of 156.3 with p-value of 9.24e-79, and p-values of 0.000, 0.022, and 0.013, respectively, for the three regressors. While our original data exhibited skew that led to a somewhat asymmetric residual distribution, there was no observed heteroscedasticity, nonlinear patterns, or serious outliers that would argue against the selection of a linear model. 
# 
# Overall, the above investigation serves as a good illustration of the difference between significance and usefulness. The relationships identified by regression were certainly significant, but if model effectiveness were a priority, the most desirable option would be to gather more predictive data. For example, the pure rating data is more predictive than our derived features, with an adjusted R-squared of 0.629 even before aggregation:

# In[15]:


original['intercept'] = np.ones(len(original))
original_ratings = original[['appearance',
                             'aroma',
                             'palate',
                             'taste',
                             'intercept']]
original_Y = original['overall']
reg_original = sm.OLS(original_Y, original_ratings).fit()
y_pred_original = reg_original.predict(original_ratings)

plt.scatter(original_Y, y_pred_original)
plt.xlabel("Ratings: $Y_i$")
plt.ylabel("Predicted ratings: $\hat{Y}_i$")
plt.title("Ratings vs Predicted ratings: $Y_i$ vs $\hat{Y}_i$")

print(reg_original.summary())


# Sometimes one finds fascinating relationships in data sources that do not initially appear connected, and this is the type of data science journalists love to write stories about. The rest of the time, the data that looks connected likely is and that which doesn't isn't. Nonetheless, we have learned at least one non-obvious lesson: all else being equal, young women appear to look more kindly on the beer they drink.
