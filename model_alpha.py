import opendatasets as od
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



dataset_url = 'https://www.kaggle.com/datasets/georgescutelnicu/byzantine-emperors-3301453'
od.download(dataset_url)

# Load the dataset
file_path = '/content/byzantine-emperors-3301453/ByzantineEmperors.csv'
df = pd.read_csv(file_path)

columns_to_convert = ['Year_of_birth', 'Year_of_death', 'Start_of_reign', 'End_of_reign']


df.replace('?', np.nan, inplace=True)
df.dropna(subset=columns_to_convert, inplace=True)


df[columns_to_convert] = df[columns_to_convert].astype(float)

# Feature Engineering
df['Reign_duration'] = df['End_of_reign'] - df['Start_of_reign']
df['Log_Reign_Duration'] = np.log1p(df['Reign_duration'])
df['Age_at_Death'] = df['Year_of_death'] - df['Year_of_birth']
df['Age_at_Start_of_Reign'] = pd.to_datetime(df['Start_of_reign'], errors='coerce').dt.year - df['Year_of_birth']
df['Reign_Per_Year_Start'] = df['Reign_duration'] / df['Age_at_Start_of_Reign']
df['Avg_Dinasty_Duration'] = df.groupby('Dinasty')['Reign_duration'].transform('mean')
df['Time_to_Reign'] = pd.to_datetime(df['Start_of_reign'], errors='coerce').dt.year - df['Year_of_birth']
df['Time_to_End'] = pd.to_datetime(df['End_of_reign'], errors='coerce').dt.year - df['Year_of_death']

# Drop unnecessary columns including datetime columns
df = df[['Dinasty', 'Location_of_birth', 'Cause_of_death', 'Reign_duration', 'Log_Reign_Duration',
         'Age_at_Death', 'Age_at_Start_of_Reign', 'Reign_Per_Year_Start', 'Avg_Dinasty_Duration',
         'Time_to_Reign', 'Time_to_End', 'Name']]


df_encoded = pd.get_dummies(df, columns=['Dinasty', 'Location_of_birth', 'Cause_of_death'], drop_first=True)


features = df_encoded.drop(['Reign_duration', 'Log_Reign_Duration', 'Name'], axis=1)
target = df_encoded['Reign_duration']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')


history = model.fit(X_train_scaled, y_train, 
                    epochs=200, batch_size=32, 
                    validation_data=(X_test_scaled, y_test), 
                    verbose=1)


y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error on Test Data: {mse}')
print(f'Mean Absolute Error on Test Data: {mae}')
print(f'R-squared on Test Data: {r2}')

predictions_with_names = pd.DataFrame({'Actual_Reign_Duration': y_test, 'Predicted_Reign_Duration': y_pred.flatten()})
print(predictions_with_names)



plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
