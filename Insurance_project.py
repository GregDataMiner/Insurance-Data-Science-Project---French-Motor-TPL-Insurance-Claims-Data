# Import key modules that will be used throughout the project.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs/plotting

MTPL_filepath1 = ".../freMTPLfreq.xlsx"
MTPL_filepath2 = ".../freMTPLsev.xlsx"

print("Now loading MTPLfreq.")
MTPLfreq = pd.read_excel(MTPL_filepath1)
print("MTPLfreq was loaded./n")

print("Now loading MTPLsev.")
MTPLsev = pd.read_excel(MTPL_filepath2)
print("MTPLsev was loaded./n")

# Check for total amount of claims paid in original DataFrame, prior to merging MTPLfreq with MTPLsev.
print(sum(MTPLsev['ClaimAmount']))

# Aggregate the claim amounts by PolicyID, prior to merging MTPLfreq with MTPLsev.
MTPLsev_grp = MTPLsev.groupby(['PolicyID'])[['ClaimAmount']].agg('sum').reset_index()

# Perform an outer merge between MTPLfreq/MTPLsev, based on PolicyID, then reset the index back to PolicyID (this is dropped during merging).
df_merged = pd.merge(MTPLfreq, MTPLsev_grp, how='outer', on='PolicyID').fillna(0).set_index('PolicyID')

# Check for the total amount of claims paid in new DataFrame, after merging MTPLfreq with MTPLsev.
print(sum(df_merged['ClaimAmount']))


print(df_merged.columns)
print('\n')
print(df_merged.dtypes)
print('\n')
print(df_merged.head())
print('\n')


policies_no_claims = len(df_merged.loc[df_merged['ClaimNb'] == 0])
all_policies = len(df_merged.index)

pct_pols_no_clm = round((policies_no_claims/all_policies)*100, 2) 

print(str(pct_pols_no_clm)+"% of policyholders have not made any claims.")

"""
Générer des caractéristiques supplémentaires basées sur les interactions/transformations
 des variables existantes
"""
df_merged['ClaimFreq'] = df_merged['ClaimNb'] / df_merged['Exposure']

df_merged['ClaimSev'] = df_merged['ClaimAmount'] / df_merged['Exposure']

"""
Les fuites de données 
Les fuites de données dans l'apprentissage automatique se produisent lorsqu'un 
modèle utilise des informations pendant la formation qui ne seraient pas disponibles 
au moment de la prédiction. Les fuites font qu'un modèle prédictif semble précis 
jusqu'à ce qu'il soit déployé dans son cas d'utilisation ; ensuite, il produira 
des résultats inexacts, ce qui entraînera une mauvaise prise de décision et de fausses idées.

Echantillionage 
La plupart des modeles de machine learning necessitent une chantillon d'apprentissage, 
et une chantillon test. Ceci est d'autant plus vrai que l'objectif du modele est 
la prediction ou le classement. Le jeu de donnees detudes doit donc etre partitionne 
en un echantillon d'entrainement du modele et une echantillon test pour evaluer ses performances.
"""

# Affecter la variable cible à son propre cadre de données.
y_full = df_merged.ClaimAmount

# Attribuer les caractéristiques à leur propre cadre de données. 
# Supprimez également ClaimSev, afin d'éviter les fuites de données lors de la prédiction de ClaimAmount.
X_full = df_merged.drop(['ClaimAmount', 'ClaimSev'], axis=1)

print("Target variable (y_full) preview:")
print(y_full.head())
print("\nFeature set (X_full) preview:")
print(X_full.head())

from sklearn.model_selection import train_test_split

# Fonction de division train/test personelle, on ne l'utilise pas dans ce projet 
def split_train_test(data, test_ratio):
    # Mélanger les indices
    shuffled_indices = np.random.permutation(len(data))
    
    # Calculer la taille du set de test
    test_set_size = int(len(data) * test_ratio)
    
    # Séparer les indices en train/test
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # Retourner les sous-ensembles de données
    return data.iloc[train_indices], data.iloc[test_indices]
# Exemple d'utilisation de la fonction personnalisée (optionnel)
# X_train_custom, X_test_custom = split_train_test(X_full, 0.2)


# Utilisation de `train_test_split` de sklearn, plus simple
X_train, X_valid, y_train, y_valid = train_test_split(
    X_full, y_full, train_size=0.8, test_size=0.2, random_state=1
)

print("\nTraining set (X_train) preview:")
print(X_train.head())

print("\nValidation set (X_valid) preview:")
print(X_valid.head())


"""
Label Encoding

Here, we label-encode the Power column such that it changes each (ordinal) 
text-based label to a numerical value which is machine-interpretable, for later 
use in feature scaling as well as model fitting.
"""

from sklearn.preprocessing import LabelEncoder

# Make a copy of the training/validation feature subsets to avoid changing any original data.
copy_X_train = X_train.copy()
copy_X_valid = X_valid.copy()

# Apply a label encoder to the 'Power' column (i.e. encoding of ordinal variable).
label_encoder = LabelEncoder()

copy_X_train['Power'] = label_encoder.fit_transform(X_train['Power'])
copy_X_valid['Power'] = label_encoder.transform(X_valid['Power'])



"""
One-Hot Encoding

We also one-hot encode the Brand, Gas and Region columns, such that these 
categories are converted to numerical and machine-interpretable values that can
 be supplied to each regression model.
"""

from sklearn.preprocessing import OneHotEncoder

# Initialise a one-hot encoder to columns that contain categorical data.
OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
OH_cols = ['Brand', 'Gas', 'Region']

## We set handle_unknown='ignore' to avoid errors when the validation data contains classes that aren't represented
## in the training data, and setting sparse=False ensures that the encoded columns are returned as a numpy array
## (instead of a sparse matrix).

# Use the one-hot encoder to transform the categorical data columns. 
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(copy_X_train[OH_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(copy_X_valid[OH_cols]))

# One-hot encoding removes the index; re-assign the original index.
OH_cols_train.index = copy_X_train.index
OH_cols_valid.index = copy_X_valid.index

# Add column-labelling back in, using the get_feature_names_out() function. 
OH_cols_train.columns = OH_encoder.get_feature_names_out(OH_cols)
OH_cols_valid.columns = OH_encoder.get_feature_names_out(OH_cols)

# Create copies that only include numerical feature columns (these will be replaced with one-hot encoded versions).
copy_X_train_no_OH_cols = copy_X_train.drop(OH_cols, axis=1)
copy_X_valid_no_OH_cols = copy_X_valid.drop(OH_cols, axis=1)

# Concatenate the one-hot encoded columns with the existing numerical feature columns.
X_train_enc = pd.concat([copy_X_train_no_OH_cols, OH_cols_train], axis=1)
X_valid_enc = pd.concat([copy_X_valid_no_OH_cols, OH_cols_valid], axis=1)

"""
Data scaling - normalisation

Next, we perform min-max scaling on the encoded dataset, such that all features
 lie between 0 and 1 - this is so that, when training any of the regression models, 
 all features will have variances with the same order of magnitude as each other.
 Thus, no single feature will dominate the objective function and prohibit the 
 model from learning from other features correctly as expected.
 
"""
 
from sklearn.preprocessing import MinMaxScaler

# Initialise the MinMaxScaler model, then fit it to the (encoded) training feature dataset.
MM_scaler = MinMaxScaler()
MM_scaler.fit(X_train_enc)

# Fit the scaler, then normalise/transform both the training and validation feature datasets.
X_train_scale = pd.DataFrame(MM_scaler.transform(X_train_enc), index=X_train_enc.index,
                             columns=X_train_enc.columns)

X_valid_scale = pd.DataFrame(MM_scaler.transform(X_valid_enc), index=X_valid_enc.index, 
                             columns=X_valid_enc.columns)
 
#Here, we check to ensure that all feature values are now numerically encoded and are between 0 and 1.
# Verify minimum value of all features in X_train_scale:

X_train_scale.min(axis=0)
X_train_scale.mean(axis=0)
X_train_scale.max(axis=0)
X_valid_scale.min(axis=0)
X_valid_scale.mean(axis=0)
X_valid_scale.max(axis=0)


#Here, we use the pd.describe() function to obtain descriptive statistics of 
#the original dataset (prior to preprocessing).
print(df_merged.describe())

"""
Generate pairplots between the targets and features to understand the relationships
 between them and discover whether there are any trends/correlations within the data.
To do this, we will use the seaborn.pairplot() function as a high-level interface 
to plot the pairwise relationships in the df_merged dataset.

Plot pairwise relationships in a dataset.

By default, this function will create a grid of Axes such that each numeric 
variable in data will by shared across the y-axes across a single row and 
the x-axes across a single column. The diagonal plots are treated differently: 
a univariate distribution plot is drawn to show the marginal distribution 
of the data in each column. It is also possible to show a subset of variables 
or plot different variables on the rows and columns.

First, we define two separate lists of x-variables that we will produce pairplots with, 
depending on which y-variable we choose.
"""
import seaborn as sns
desc_pairplot_x_vars_A = ['ClaimNb', 'Power', 'CarAge', 'DriverAge', 'Brand', 'Gas', 'Density']
desc_pairplot_x_vars_B = ['Exposure','Power', 'CarAge', 'DriverAge', 'Brand', 'Gas', 'Density']
desc_pairplot_1 = sns.pairplot(df_merged, x_vars=desc_pairplot_x_vars_A, y_vars='Exposure')
desc_pairplot_1 = sns.pairplot(df_merged, x_vars=desc_pairplot_x_vars_B, y_vars='ClaimNb')
desc_pairplot_x_vars_C = ['Power', 'CarAge', 'DriverAge', 'Brand', 'Gas', 'Density']
desc_pairplot_3 = sns.pairplot(df_merged, x_vars=desc_pairplot_x_vars_C, y_vars='ClaimFreq')

# Pairplot 4 - ClaimAmount vs. x_vars.
desc_pairplot_4 = sns.pairplot(df_merged, x_vars=desc_pairplot_x_vars_B, y_vars='ClaimAmount')

# Pairplot 5 - ClaimSev vs. x_vars (i.e. accounting for policy exposure weighting).
desc_pairplot_5 = sns.pairplot(df_merged, x_vars=desc_pairplot_x_vars_A, y_vars='ClaimSev')


"""
Step 8: Perform feature selection via L1 regularisation
Next, we will perform feature selection via L1 (lasso) regularisation, in order 
to reduce the number of features that are used for fitting each of the models - 
this is done in order to prevent overfitting. To do this, we add a regularisation 
term (containing the L1 norm) to the standard loss function that is to be minimised, such that:
    
    Loss= Error(y,^y)+λSwi
    
  y is the true value/severity of the claim
  ^y is the claim value/severity predicted by the model 
  λ>0 is the regularisation parameter that determines the strength of regularisation to be applied to the loss function 
  wi is the weight of feature i

This modified loss function is then subsequently minimised in order to produce 
the parameters of the Lasso linear regression model. Features that are less 
significant in producing the Lasso model will have their weights/importances 
decreased towards 0 - these "unimportant" features can then be removed from 
the set of inputs/features that are supplied to the models we will use later on.
    
"""
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Establish the Lasso (L1) Regularisation model that will perform feature selection.
lasso = Lasso(alpha=1e-4, random_state=1).fit(X_train_scale, y_train)
model = SelectFromModel(lasso, prefit=True)

X_train_l1 = model.transform(X_train_scale)

selected_features = pd.DataFrame(model.inverse_transform(X_train_l1),
                                index=X_train_scale.index,
                                columns=X_train_scale.columns)

print(selected_features)
selected_columns = selected_features.columns[selected_features.var() != 0]
print(selected_columns)
# retaraitement des colones prime 'Brand_Volkswagen Audi, Skoda or Seat'


X_train_L1reg = selected_features.drop(selected_features.columns[selected_features.var() == 0], axis=1)

print(X_train_L1reg.columns)
print(X_train_L1reg)
# The X_valid dataframe is truncated such that only the L1-selected features are used for validation purposes.
X_valid_L1reg = X_valid_scale[selected_columns]

"""
Step 9: Define the regressors/models used

In this project, we aim to predict the target (ClaimAmount) using the following linear regression approaches:

    Random Forest Regression
They train multiple decision trees on random subsamples of the dataset.
The predictions of each tree are averaged to reduce overlearning and improve robustness.
The principle of Random Forest is quite similar to bagging, except that the drills each time change the very structure of the model.
the exception that the drills change the very structure of the model each time by
 selecting, in addition to the subsample, a subset of the covariates x. We
 We then have a sequence of different trees (drills) trained on different subsamples.
 samples.

Poisson Regression (GLM)
Generalized Linear Model (GLM) assuming that the target variable follows a Poisson distribution.
Used to model continuous positive claims, but better alternatives exist for this type of data.

Tweedie Regression
GLM model based on a Tweedie distribution (a mixture of Poisson and Gamma).
Handles claims with a high concentration of zero values (no claims) and a long tail for high claims.

XGBoost Regression
Gradient boosting method that improves predictions by successively building decision trees on the residuals of previous predictions.
Highly effective thanks to gradient descent optimisation and widely used in regression problems.


"""



"""
Step 10: Perform cross-validation to obtain the optimal set of hyperparameters for each model¶

Here, we will perform 5-fold cross-validation in order to optimise one of each models' hyperparameters. These are:

RandomForestRegressor

n_estimators represents the number of decision trees that are implemented by the random forest regressor. We will aim to optimise this hyperparameter.
random_state sets the random number seed and is used for reproducibility purposes. Here, we set this value to 1.
n_jobs represents the number of calculations to run in parallel; setting a value of -1 means that all processors will be used.
PoissonRegressor

alpha represents the constants that multiplies the penalty term, thus determining the strength of regularisation for the Poisson GLM used. We will aim to optimise this hyperparameter.
max_iter represents the maximal number of iterations for the PoissonRegressor's solver.
TweedieRegressor

power determines the underlying target value's distribution - using a value between 1 and 2 produces a compound Poisson-Gamma distribution.
As a pure Gamma distribution's probability density is not defined at x=0, we set this value to 1.8 such that the target's compound distribution shows more Gamma form than Poisson. This is another hyperparameter that could potentially be optimised for simultaneously, via grid-search methods.

alpha represents the constants that multiplies the penalty term, thus determining the strength of regularisation for the Tweedie GLM used. We will aim to optimise this hyperparameter.
max_iter represents the maximal number of iterations for the TweedieRegressor's solver.
XGBRegressor

n_estimators represents the number of gradient boosted trees implemented by the eXtreme Gradient Boosting (XGB) regressor; this is equivalent to the number of boosting rounds. We will aim to optimise this hyperparameter.
learning_rate refers to the boosting learning rate/step size of the XGB regressor - this value is between 0 and 1.
random_state sets the random number seed and is used for reproducibility purposes. Here, we set this value to 1.

"""


# Import the regression models from sklearn/xgboost.
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import TweedieRegressor
from xgboost import XGBRegressor

# Import the cross_val_score function from sklearn.
from sklearn.model_selection import cross_val_score


## Define scoring functions for each method.

def get_score_RF(n_estimators):
    model_RF = RandomForestRegressor(n_estimators=n_estimators, random_state=1, n_jobs=-1)
    
    scores_RF = -1 * cross_val_score(model_RF, X_train_L1reg, y_train,
                              cv=5,
                              scoring='neg_mean_absolute_error')

    return scores_RF.mean()


def get_score_PGLM(alpha):
    model_PGLM = PoissonRegressor(alpha=alpha, max_iter=500)
    
    scores_PGLM = -1 * cross_val_score(model_PGLM, X_train_L1reg, y_train,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    
    return scores_PGLM.mean()


def get_score_TGLM(alpha):
    model_TGLM = TweedieRegressor(power=1.8, alpha=alpha, max_iter=500)
    
    scores_TGLM = -1 * cross_val_score(model_TGLM, X_train_L1reg, y_train,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    
    return scores_TGLM.mean()


def get_score_XGB(n_estimators):
    model_XGB = XGBRegressor(n_estimators=n_estimators,
                               learning_rate=0.01,
                               random_state=1)
    
    scores_XGB = -1 * cross_val_score(model_XGB, X_train_L1reg, y_train,
                                     cv=5,
                                     scoring='neg_mean_absolute_error')
    
    return scores_XGB.mean()


## Create empty dictionaries which will be used to store the scoring results for each method.

results_RF = {}
results_PGLM = {}
results_TGLM = {}
results_XGB = {}


## Obtain 8 scores for the RandomForestRegressor model.

for i in range(1, 9):
    results_RF[100*i] = get_score_RF(100*i)
    print("results_RF{} recorded".format(i))

print("RF done")


## Obtain 8 scores for the PoissonRegressor model.

for i in range(1, 9):
    results_PGLM[round(0.2*i, 2)] = get_score_PGLM(round(0.2*i, 2))
    print("results_PGLM{} recorded".format(i))

print("PGLM done")


## Obtain 8 scores for the TweedieRegressor model.

for i in range(1, 9):
    results_TGLM[round(0.01*i, 2)] = get_score_TGLM(round(0.01*i, 2))
    print("results_TGLM{} recorded".format(i))

print("TGLM done")

## Obtain 8 scores for the XGBRegressor model.

for i in range(1, 9):
    results_XGB[5*i] = get_score_XGB(5*i)
    print("results_XGB{} recorded".format(i))
    
print("XGB done")
RF_n_estimators_best = min(results_RF, key=results_RF.get)
print(RF_n_estimators_best)
PGLM_alpha_best = min(results_PGLM, key=results_PGLM.get)
print(PGLM_alpha_best)

TGLM_alpha_best = min(results_TGLM, key=results_TGLM.get)
print(TGLM_alpha_best)
XGB_n_estimators_best = min(results_XGB, key=results_XGB.get)
print(XGB_n_estimators_best)


"""
Step 11: Train (fit) the models to the entire training dataset
"""
# Define the optimised regression models that will be used.

model_RF_opt = RandomForestRegressor(n_estimators=RF_n_estimators_best, random_state=1, n_jobs=-1)

model_PGLM_opt = PoissonRegressor(alpha=PGLM_alpha_best, max_iter=500)

model_TGLM_opt = TweedieRegressor(power=1.8, alpha=TGLM_alpha_best, max_iter=500)

model_XGB_opt = XGBRegressor(n_estimators=XGB_n_estimators_best, learning_rate=0.01, random_state=1)

# Fit the optimised models to the full (pre-processed) training dataset.

model_RF_opt.fit(X_train_L1reg, y_train)
print("model_RF_opt trained")

model_PGLM_opt.fit(X_train_L1reg, y_train)
print("model_PGLM_opt trained")

model_TGLM_opt.fit(X_train_L1reg, y_train)
print("model_TGLM_opt trained")

model_XGB_opt.fit(X_train_L1reg, y_train)
print("model_XGB_opt trained")


"""
Step 12: Generate a unique set of predictions for each model
The next step is to generate predictions of ClaimAmount for each policyholder 
within the pre-processed validation dataset; this is done using the .predict() 
function for each of the optimised models.

"""

# Use the trained models to generate unique sets of predicted y-values i.e. ClaimAmount.

preds_RF = model_RF_opt.predict(X_valid_L1reg)
preds_PGLM = model_PGLM_opt.predict(X_valid_L1reg)
preds_TGLM = model_TGLM_opt.predict(X_valid_L1reg)
preds_XGB = model_XGB_opt.predict(X_valid_L1reg)
print("All predictions generated")

"""
Step 13: Assess the chosen models' performance, using validation data¶

In order to evaluate and rank the models based on their regression performances, 
an appropriate scoring metric should be used. One common example of this is to 
calculate the Mean Absolute Error (MAE) for each of the fitted models against the 
validation data, which can then be ranked in order to determine the model with the 
lowest MAE, which is deemed to be the best model in terms of accuracy and goodness 
of fit:
    
    
    
However, in order to calculate the RMSE of a model, each prediction error must be squared before they are averaged together; this means that larger errors/outliers are more strongly penalised than smaller errors. Therefore, as the vast majority (~96%) of policyholders within the freMTPL dataset have not made any claims whatsoever, we do not wish to heavily penalise each of the models based on any severe claims/outliers that are incorrectly predicted, as this would increase the risk of overfitting each model to these outliers (i.e. by encouraging the model to predict large claims more frequently).
Hence, we will use the mean_absolute_error() function from sklearn.metrics to calculate the MAE score, comparing each model's predictions against the validation dataset's (true) values.
"""

from sklearn.metrics import mean_absolute_error

# Calculate the Mean Absolute Error metric for each set of predicted y-values.

MAE_RF = mean_absolute_error(y_valid, preds_RF)
MAE_PGLM = mean_absolute_error(y_valid, preds_PGLM)
MAE_TGLM = mean_absolute_error(y_valid, preds_TGLM)
MAE_XGB = mean_absolute_error(y_valid, preds_XGB)
print("All MAE scores calculated")


"""
Step 14: Evaluate the models' performances¶

"""


# Collect all MAE scores in a single dictionary.

MAE_results = {'RF': MAE_RF,
                'PGLM': MAE_PGLM,
                'TGLM': MAE_TGLM,
                'XGB': MAE_XGB}

print(MAE_results)


# Select the model with the smallest MAE.

best_model = min(MAE_results, key=MAE_results.get)
print(best_model)


"""
## Areas for Improvement in the Project
### Approach to Claim Severity Prediction

 - Currently, the project predicts the total loss amount for each policyholder without modeling the average loss per claim.
 - Derive additional features such as the maximum and minimum claim amounts per policyholder to better model severity.
 - The project also uses the actual number of claims from the dataset instead of predicting claim frequency first.
 - Implement the full frequency-severity approach by: Predicting the number of claims using a Poisson regression. Predicting the average claim severity for policyholders with non-zero claims using a Gamma-based GLM.

### Model Training and Testing
Models were trained and tested only once using a basic split (train/test).
Perform multiple training iterations with cross-validation to achieve a more robust fit to the data.

### Data Preprocessing
Categorical encoding methods may not be optimal. For example, one-hot encoding was applied to high-cardinality columns like Region, which is not recommended.
Explore alternatives such as target encoding or embedding layers to handle high-cardinality features more efficiently.

### Hyperparameter Optimization
The current approach involves optimizing one hyperparameter at a time while keeping others constant.
Implement GridSearchCV to perform comprehensive hyperparameter optimization, iterating over a grid of possible values to find the optimal combination.
Example for TweedieRegressor:

### Feature Selection
No strict feature selection was performed, increasing the risk of overfitting.
Use Lasso (L1 regularization) to reduce the number of features by setting higher values for the alpha parameter.
This would require additional hyperparameter tuning to balance feature reduction with model performance.

### For the next steps I am currently working on implementing these improvements. 

Specifically, I am enhancing the frequency-severity approach by predicting both claim frequency and severity.
Applying cross-validation to improve model robustness.
Exploring hyperparameter optimization using GridSearchCV.
Refining data preprocessing and feature selection to reduce model complexity and overfitting.
These changes aim to provide more accurate and generalizable predictions for actuarial pricing models.

"""