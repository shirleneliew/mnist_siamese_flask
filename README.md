# mnist_siamese_flask
REST API that predict the similarity of 2 MNIST inputs on Flask. Inputs are arrays in a JSON file.

## Install all requirements
`pip3 install -r requirements.txt`

## Run server
`python run_keraer.py`

## Make predictions 
Go to data folder.
`cd data`

Use curl to POST the json file to the server.
`curl -X POST -H "Content-Type: application/json" --data @mnist_same.json 'http://localhost:5000/predict'`

