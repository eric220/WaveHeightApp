import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

#Direction columns
cols = ['GSTkts', 'WSPDkts', 'WDIR_E', 'WDIR_ENE', 'WDIR_ESE', 'WDIR_N',
        'WDIR_NE', 'WDIR_NNE', 'WDIR_NNW', 'WDIR_NW', 'WDIR_S', 'WDIR_SE',
        'WDIR_SSE', 'WDIR_SSW', 'WDIR_SW', 'WDIR_W', 'WDIR_WNW', 'WDIR_WSW']

#These are the coefficients and offset for the linear model
decoded = [[0.3289281156858399, -0.17224284413490162, 0.3661848578856898, 0.3913752037623029, 0.41187221426112997, -0.524737113205118, 0.42556462169096576, -0.9218888551567251, -0.3054422200041307, 0.40619738963267427, -0.4042559807288526, -0.12618065785522273, -0.30495940939300703, -0.18842234272212566, -0.4214295964403609, 0.7716820410411276, 0.2725020585995589, 0.15193778863244567], 1.5024941576302195]

app = Flask(__name__)
model = LinearRegression()
model.coef_ = np.array(decoded[0])
model.intercept_ = np.array(decoded[1])


@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    print('FEATURES', features[0])
    df2 = pd.DataFrame([np.zeros(len(cols))], columns= cols)
    df2.at[0, 'GSTkts'] = features[0]
    df2.at[0, 'WSPDkts'] = features[1]
    direction = str(features[2]).upper()
    if 'WDIR_{}'.format(direction) in cols:
        df2.at[0, 'WDIR_{}' .format(direction)] = 1
    else:
        df2.at[0, 'WDIR_{}' .format('ENE')] = 1
    
    prediction = model.predict(df2)
    output = round(prediction[0], 3)
    return render_template('index.html', prediction_text='Waves are Predicted to be {}ft'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)