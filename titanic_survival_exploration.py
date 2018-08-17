# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Set a random seed
import random
random.seed(42)

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())


# Recall that these are the various features present for each passenger on the ship:
# - **Survived**: Outcome of survival (0 = No; 1 = Yes)
# - **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
# - **Name**: Name of passenger
# - **Sex**: Sex of the passenger
# - **Age**: Age of the passenger (Some entries contain `NaN`)
# - **SibSp**: Number of siblings and spouses of the passenger aboard
# - **Parch**: Number of parents and children of the passenger aboard
# - **Ticket**: Ticket number of the passenger
# - **Fare**: Fare paid by the passenger
# - **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
# - **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
#
# Since we're interested in the outcome of survival for each passenger or crew member, we can remove the **Survived** feature from this dataset and store it as its own separate variable `outcomes`. We will use these outcomes as our prediction targets.
# Remove **Survived** as a feature of the dataset and store it in `outcomes`.

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(features_raw.head())

#No channges in the original dataset
display(full_data.head())


# The very same sample of the RMS Titanic data now shows the **Survived**
# feature removed from the DataFrame. Note that `data` (the passenger data)
# and `outcomes` (the outcomes of survival) are now *paired*.
# That means for any passenger `data.loc[i]`, they have the survival
# outcome `outcomes[i]`.
#
# ## Preprocessing the data
#
# Now, let's do some data preprocessing. First, we'll one-hot encode the features.

features = pd.get_dummies(features_raw)


# And now we'll fill in any blanks with zeroes.

features = features.fillna(0.0)
display(features.head())


# Training the model
#
# Now we're ready to train a model in sklearn.
# First, let's split the data into training and testing sets.
# Then we'll train the model on the training set.

X_train, X_test, y_train, y_test = train_test_split(features, outcomes, \
                                        test_size=0.2, random_state=42)

# Define the classifier, and fit it to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# ## Testing the model
# Now, let's see how our model does,
# let's calculate the accuracy over both the training and the testing set.

# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)


# # Exercise: Improving the model
#
# high training accuracy and a lower testing accuracy.
# We may be overfitting a bit.
#
# Train a new model, and try to specify some parameters in order to
# improve the testing accuracy, such as:
# - `max_depth`
# - `min_samples_leaf`
# - `min_samples_split`

# search for best accuracy
for i in range(20):
    for j in range(20):
        # Train the model
        model = DecisionTreeClassifier(max_depth=i+1,min_samples_leaf=j+1,min_samples_split=0.01)
        model.fit(X_train, y_train)
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # Calculate the accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        if train_accuracy >= 0.85:
            if test_accuracy >= 0.85:
                print('max_depth: ',i+1)
                print('min_samples_leaf: ',j+1)
                print('min_samples_split: 0.01')
                print('The training accuracy is', train_accuracy)
                print('The test accuracy is', test_accuracy)

# Final model

# Train the model
model = DecisionTreeClassifier(max_depth=7,min_samples_leaf=6,min_samples_split=0.01)
model.fit(X_train, y_train)
# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
#     print('max_depth: ',i+1)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
