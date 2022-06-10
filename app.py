from flask import Flask,jsonify,request
from classifier import GetPrediction

app=Flask(__name__)

@app.route("/predict-digit",methods=["POST"])
def predictdata ():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction":prediction
        
    }),200