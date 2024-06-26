# Byzantine Reing Prediction

This is a model which predicts how long Byzantine emperors will reign trained courtesy of the Byzantine Emperors (330-1453) dataset on Kaggle.

## Results of the model

<img src="graph.png" alt="Example Image" width="400" height="300" />


Above is the loss function through the traininig. The loss significantly dropped until the 250th epoch. After 250 epochs the loss steadily dropped to between 2 to 3 at the end.

| Emperor Name| Predicted Reign Length | Original Reign Length |
| --------------- | --------------- | --------------- |
| Manuel II    | 35 years | 34 years    |
| Basil II    | 49 years   | 50 years   |
| Constantine I    | 30 years   | 31 years   |
| John VII    | 3 years   | 3 years   |
| Leo VI    | 24 years   | 26 years  |
| Constantine VII    | 45 years   | 46 years   |

The table above shows some prediction made on the test data. Altough later emperors were easier to predict the model struggled with early emperors such as Constantine I due to the empire not beign quite in a stable state. Altough after the first hundred years of the establishement of the Byzantine Empire the model was generally quite accurate in it's predictions generally having an offset of three years at maximum. The struggle in the first 100-150 years likely happened because of the barbarian raids on the Eastern and Western Roman Empire.


## Feature Engineering Performed

Altough a subpar dataset was provided to be able to create a better model columns had to be derived from the given data columns. The columns created were: Reign_Duration, Log_Reign_Duration, Age_at_Death, Age_at_Start_of_Reign, Reign_Per_Year_Start, Avg_Dinasty_Duration, Time_to_Reign, Time_to_End. Through the creation of these columns the loss was reduced from 35-30 to 2-3 providing a significant benefit to the model. 

## Features used to train the model

Dinasty: This categorical feature represents the dynasty of the emperor. It has been one-hot encoded.
Location_of_birth: Another categorical feature indicating the birth location of the emperor. Also, it has been one-hot encoded.
Cause_of_death: Categorical feature indicating the cause of death of the emperor. One-hot encoded as well.
Age_at_Death: The age of the emperor at the time of death.
Age_at_Start_of_Reign: The age of the emperor at the start of their reign.
Reign_Per_Year_Start: The duration of reign per year at the start of the reign.
Avg_Dinasty_Duration: The average duration of reign within the same dynasty.
Time_to_Reign: Time taken to assume the reign.
Time_to_End: Time taken from the year of birth to the end of reign.
