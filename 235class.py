import pandas as pd
import pickle
from tensorflow.keras.models import model_from_json
from sklearn.neural_network import MLPClassifier

dataset = pd.read_csv('234_new_radar_distance_data.csv')

x = dataset.iloc[:, [2,3, 4]].values # input
  
y = dataset.iloc[:, 5].values # output

#CNN model
model = MLPClassifier(hidden_layer_sizes= (20), 
						random_state=5, 
						activation='relu', 
						batch_size=200, 
						learning_rate_init=0.03) 
model.fit(x, y)

#prediction
predictions = model.predict(x)
print("Predicted data:",predictions)

# save the classifier
file = open('model.pkl', 'wb')
pickle.dump(model, file) 
print(file)   
# load it again
saved_model_file = open('model.pkl', 'rb')
saved_model = pickle.load(saved_model_file)
print(saved_model)  
#test model
car_data = {
    'throttle': [0.364332455],
    'steer':[0.654],
    'distance': [6.674762726]
}
data = pd.DataFrame(car_data, columns= ['throttle','steer', 'distance'])
predicated_data = saved_model.predict(data)

print("Your prediction from model is: ", predicated_data)
