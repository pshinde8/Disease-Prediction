import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

model = pickle.load(open("rf_model.pkl", "rb"))

data = pd.read_csv('dataset/disease_dataset.csv')
data_sevrity = pd.read_csv('dataset/Symptom-severity.csv')
symp_description = pd.read_csv('dataset/symptom_Description.csv')
symp_precaution = pd.read_csv('dataset/symptom_precaution.csv')

X = data.drop(['Disease'],axis=1)
db = X.columns.values

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == "POST":
        features = [str(x) for x in request.form.getlist('symptoms')]
        print("features -", features)

        new_db = [0 for i in range(len(db))]
        print(new_db)

        for i, symp in enumerate(features):
            for index, row in data_sevrity.iterrows():
                if row['Symptom'] == symp:
                    corresponding_value = row['weight']
                    new_db[i] = corresponding_value
                    break

        print(new_db)
        final = np.array(new_db).reshape(1,-1)
        print("final - ",final)

        pred = model.predict(final)
        print("pred -", pred)

        if pred:
            predicted_disease = pred[0].replace('_', ' ')
        else:
            predicted_disease = "Unknown"

        dis = predicted_disease
        print(dis)

        description=symp_description.loc[symp_description['Disease'] == dis, 'Description'].values[0]
        print(description)

        precautions=symp_precaution.loc[symp_precaution['Disease']==dis, ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.tolist()[0]
        print(precautions)

    return render_template("submit.html", pred = predicted_disease, descri = description, precau = precautions)

if __name__ == "__main__":
    app.run(debug=True, use_reloader = True)
