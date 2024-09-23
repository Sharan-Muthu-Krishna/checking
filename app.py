from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('music-recommender.joblib')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict_genre():
    data = request.get_json()

    age = data['age']
    gender = data['gender']

    # Prepare input data
    input_data = pd.DataFrame([[age, gender]], columns=['age', 'gender'])

    # Make prediction
    genre_encoded = model.predict(input_data)[0]

    # Decode the predicted genre
    genre = label_encoder.inverse_transform([genre_encoded])[0]

    # Return the prediction as a JSON response
    return jsonify({
        'age': age,
        'gender': gender,
        'predicted_genre': genre
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify(message="Hello, I am Animesh!")

if __name__ == '__main__':
    app.run(debug=True)
