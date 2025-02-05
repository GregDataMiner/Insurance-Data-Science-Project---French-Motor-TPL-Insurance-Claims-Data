# Import key modules that will be used throughout the project.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs/plotting

MTPL_filepath1 = "C:/Users/grego/Documents/USPN M2/REASSURANCE/Projet/freMTPLfreq.xlsx"
MTPL_filepath2 = "C:/Users/grego/Documents/USPN M2/REASSURANCE/Projet/freMTPLsev.xlsx"







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
lasso = Lasso(alpha=5e-5, random_state=1, max_iter=1e+6).fit(X_train_scale, y_train)
model = SelectFromModel(lasso, prefit=True)

X_train_l1 = model.transform(X_train_scale)

selected_features = pd.DataFrame(model.inverse_transform(X_train_l1),
                                index=X_train_scale.index,
                                columns=X_train_scale.columns)

print(selected_features)






