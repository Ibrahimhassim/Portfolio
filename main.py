# Videos that were used for guidance
# https://www.youtube.com/watch?v=MtWt39aEMxo
# https://www.google.com/search?sca_esv=807b745412b14cf8&sxsrf=ACQVn0_8K55YguLc8EHIflCA0qdv_rLOVg:1712586085954&q=how+do+i+used+mean+squared+error+for+my+linear+regression+model+in+pycharm&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwjDrbOq6LKFAxVKVvEDHfO0AO8Q0pQJegQIEBAB&biw=1396&bih=639&dpr=1.38#fpstate=ive&vld=cid:263fd93c,vid:y7GzO7ZB0YA,st:0
# https://www.google.com/search?sca_esv=807b745412b14cf8&sxsrf=ACQVn0-3OzE7Led-psnuF1ZE1HGpfhsfWQ:1712585915323&q=how+do+i+use+sklearn+for+cross+validation+on+my+linear+regression+model+in+pycharm&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwiN4YTZ57KFAxWPefEDHYYfBPAQ0pQJegQIDxAB&biw=1396&bih=639&dpr=1.38#fpstate=ive&vld=cid:9b6db351,vid:eTkAJQLQMgw,st:0
# https://www.google.com/search?sca_esv=807b745412b14cf8&sxsrf=ACQVn0_AGiHV-6JpDZXx3Km0jzXTZaRb-w:1712586557504&q=how+do+i+use+ridge+regularization+in+my+linear+regression+model+using+the+model.fit+function+in+pycharm&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwiTvaCL6rKFAxW4Q_EDHZojDscQ0pQJegQIERAB&biw=1229&bih=562&dpr=1.56#fpstate=ive&vld=cid:0278863d,vid:c3SkmBZ0HZw,st:0
# https://www.google.com/search?sca_esv=582779246&sxsrf=AM9HkKmaUI4VMpG7T3pSUrUocmsaZj4pxw:1700087391733&q=how+do+i+import+a+datafile+in+pycharm&tbm=vid&source=lnms&sa=X&ved=2ahUKEwjxz8iIh8eCAxWSYEEAHa9oCrgQ0pQJegQIEBAB&biw=1396&bih=650&dpr=1.38#fpstate=ive&vld=cid:4bfc21b2,vid:G4UR-EANX3w,st:0
#

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Setting up dataset
df = pd.read_csv('C:/Users/ibrah/Downloads/Salary.csv')

# This creates a scatterplot to visualize the relationship between years of experience (x)
# and salary (y).
plt.scatter(df['YearsExperience'], df['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Scatterplot of Years of Experience vs Salary')
plt.show()

# This splits the dataset into training and testing sets for model training and evaluation
x_train, x_test, y_train, y_test = train_test_split(df[['YearsExperience']], df['Salary'],
                                                    test_size=0.2, random_state=42)

# This function trains a linear regression model using the training data. It takes the training
# features x_train.
# and corresponding target y_train as input.
model = LinearRegression()
model.fit(x_train, y_train)

# This function makes predictions using the trained linear regression model
# and evaluates the model's.
y_pred = model.predict(x_test)

# This function visualizes the regression line along with the scatterplot of the data points.
plt.scatter(df['YearsExperience'], df['Salary'])
plt.plot(df['YearsExperience'], model.predict(df[['YearsExperience']]), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs Salary')
plt.show()

# Verification of results (MSE)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Cross Validation
scores = cross_val_score(model, x_train, y_train, cv=5)
print("Cross-validated R-squared scores:", scores)

# Ibrahim Make sure you follow the steps in this order when doing this again in the future.
# Making a prediction of a persons salary based on the number of years they worked.
# Step 1: Import necessary libraries (its at the top)
# Step 2: Load and preprocess the dataset (already done earlier)

# Step 3: Split the dataset into training and testing sets
X = df[['YearsExperience']].values  # Assuming 'YearsExperience' is the column name for years of experience
y = df['Salary'].values  # Assuming 'Salary' is the column name for salary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Ridge regression model
alpha = 1.0  # Regularization strength (This value is adjustable)
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)

# Step 5: Make predictions for a new data point
input_years_experience = 5  # Number of years of experience for which you want to make the prediction
input_years_experience = np.array([[input_years_experience]])  # Reshaped to a 2D array

predicted_salary = ridge_model.predict(input_years_experience)

# Step 6: Display the prediction
print("Predicted Salary for 5 years of experience:", predicted_salary)


# This model has an R - squared value of 0.89 that tells us that the goodness of the fit
# is 89% which is a very good and that 89% of all the variations observed for the dependent variable
# can be explained by the independent variables.
# The mean R squared value for my cross validation is around 0.9629 which means that
# linear regression model explains approximately 96.29% of the variance in "Salary" based on
# "Years Experience", this tells me that the model is performing well in capturing the relationship
# between the salary and the years of experience.The high R squared value also means that
# the model has a very good predictability accuracy


