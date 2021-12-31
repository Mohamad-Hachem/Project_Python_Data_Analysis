

from flask import Flask, render_template ,request
import pickle

#initialising the app
app = Flask(__name__)


#on which route we can find the app plus what are the allowed methods
@app.route('/',methods=['GET','POST'])
def home():
    prediction = 0
    if request.method == 'POST':
        model = pickle.load(open('GradientBoostingModel.pkl','rb'))
        #collecting the user input to fed them to the machine learning model
        discharge_disposition_id = int(request.form.get('discharge_disposition_id'))
        number_inpatient = int(request.form.get('number_inpatient'))
        number_diagnoses = int(request.form.get('number_diagnoses'))
        age = int(request.form.get('age'))
        number_emergency = int(request.form.get('number_emergency'))
        number_outpatient = int(request.form.get('number_outpatient'))
        num_medications = int(request.form.get('num_medications'))
        num_lab_procedures = int(request.form.get('num_lab_procedures'))
        admission_type_id = int(request.form.get('admission_type_id'))
        diabetesMed = int(request.form.get('diabetesMed'))
        diag_1 = int(request.form.get('diag_1'))
        time_in_hospital = int(request.form.get('time_in_hospital'))
        answers = [[discharge_disposition_id,number_inpatient,number_diagnoses,age,number_emergency,number_outpatient,num_medications,num_lab_procedures,admission_type_id,diabetesMed,diag_1,time_in_hospital]]
        prediction = model.predict(answers)
        if(prediction == [0]):
            print("we predict that the patient won't return to the hospital after treatment")
        if (prediction == [1]):
            print("we predict that the patient will return to the hospital after treatment in less than 30 days")
        if (prediction == [2]):
            print("we predict that the patient will return to the hospital after treatment in more than 30 days")
        #print(model.predict([[-0.49353121 ,-0.29652926 , 0.87873064 , 1.21364438 , 1.89292152 , 9.0393122,-1.40506354 ,-0.7547178  , 0.59558439, -1.77850375, -0.3057411 , -0.77710735]]))
        print("values",discharge_disposition_id,number_inpatient,number_diagnoses,age,number_emergency,number_outpatient,num_medications,num_lab_procedures,admission_type_id,diabetesMed,diag_1,time_in_hospital)

        #even tho we removed this from our training model we will keep them for future testing
        # race = int(request.form.get('race'))
        # gender = int(request.form.get('gender'))
        # num_procedures = int(request.form.get('num_procedures'))
        # diag_2 = int(request.form.get('diag_2'))
        # diag_3 = int(request.form.get('diag_3'))
        # max_glu_serum = int(request.form.get('max_glu_serum'))
        # A1Cresult = int(request.form.get('A1Cresult'))
        # metformin = int(request.form.get('metformin'))
        # glimepiride = int(request.form.get('glimepiride'))
        # glipizide = int(request.form.get('glipizide'))
        # glyburide = int(request.form.get('glyburide'))
        # rosiglitazone= int(request.form.get('rosiglitazone'))
        # change = int(request.form.get('change'))
        print("this is the model",model)
        print("this is our prediction",prediction)
    return render_template('Index.html',prediction=prediction)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
