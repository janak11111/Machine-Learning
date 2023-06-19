# import libraries
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# create flask app
app = Flask(__name__)



# load the model
# model = pickle.load(open('finalmodel.pkl', 'rb'))
# print("model >>>>>>>>>>>>>>>>>>>>>>>>>>", model)
# print(model.predict([[200, 0.0010, 0.10, 0.02, 0.0, 0.00, 0.0, 17.0, 2.1, 0.0, 100]]))





# open index page intially
@app.route('/')
def index():
	# read dataset
	df = pd.read_csv("fetal_health.csv")


	# split the dataset into train and test
	X =  df.iloc[:, :11]
	y = df.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=22, stratify=y)


	# apply svm on dataset
	model = SVC(C=50, class_weight='balanced')
	model.fit(X_train, y_train)
	return render_template('index.html')
	
	
# function to predict fetus health
@app.route('/predict', methods=['POST'])
def predict():
	# scaler = MinMaxScaler()
	print(list(request.form.values()))
	features = [[float(x) for x in list(request.form.values())]]
	print("type of features1 :", type(features))
	print(features)
	features = np.array(features)
	print("type of features2 :", type(features))
	print(features)
	# scaler.fit(features)
	f1 = minmax_scale(features)
	print("type of features2 :", type(f1))
	print(f1)
	# prediction_result = int(model.predict(f1)[0])
	# print(">>>>>>>>>>>>>>>>>", prediction_result)
	# health = ""
	# if prediction_result == 1:
		# health = "Normal"
	# elif prediction_result == 2:
		# health = "Suspect"
	# elif prediction_result == 3:
		# health = "Pathelogical"
	# , response="Your Fetus Health is : {}".format(health)
	return render_template('index.html')
	



if __name__ == "__main__":
    app.run(debug=True)