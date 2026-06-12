from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

X = np.array([[1],[2],[3],[4]])
y = np.array([2,4,6,8])

model = LinearRegression()
model.fit(X,y)

@app.get("/predict/{x}")
def predict(x: float):
    pred = model.predict([[x]])[0]
    return {"prediction": float(pred)}