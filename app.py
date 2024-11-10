from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import pandas as pd
app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: 'Normal',
1: 'Tumor',
 

}


model = load_model('kidney.h5')
 
models = load_model('kidneys.h5')
sc = pickle.load(open('StandardScaler.pk', 'rb'))

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(200,200))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 200,200,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name [classes_x[0]]

 

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		#plt.imshow(img)
		predict_result = predict_label(img_path)
		 

		#print(predict_result)
	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/chart")
def chart():
	return render_template('chart.html') 

@app.route("/performance")
def performance():
	return render_template('performance.html')  	

@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/predictions', methods = ['GET', 'POST'])
def predictions():
    return render_template('predictions.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict',methods=['POST'])
def predict():
	int_feature = [x for x in request.form.values()]
	print(int_feature)
        # Make Prediction
 
	prediction = models.predict(sc.transform(np.array([int_feature])))
	print(prediction)
        # Process your result for human
	if prediction > 0.5:
		result = "Abnormal"
	else:
		result = "Normal"
	
	return render_template('predictions.html', prediction_text= result)
@app.route('/performances')
def performances():
	return render_template('performances.html')   
	

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


