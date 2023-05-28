import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the Boston Housing dataset
dataset = fetch_openml(name='boston', version=1, as_frame=True)
X = dataset.data
y = dataset.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Create the model
model = Sequential()
model.add(Dense(1, input_shape=(X.shape[1],)))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)


# Evaluate the model on the testing data
loss = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", loss)

# Make predictions on the testing data
predictions = model.predict(X_test)

# Convert y_test to a NumPy array
y_test = np.array(y_test)

# Print some example predictions and their corresponding actual values
for i in range(5):
    print("Predicted Price:", predictions[i][0])
    print("Actual Price:", y_test[i])
    print()


