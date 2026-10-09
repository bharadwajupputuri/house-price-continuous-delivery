
import pandas as pd
import pytest

import app as house_app


DATA_PATH = "house_prices_practice.csv"
TARGET = "SalePrice"


@pytest.fixture
def client():
    house_app.app.config["TESTING"] = True

    with house_app.app.test_client() as test_client:
        yield test_client


def test_home_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json["message"] == (
        "House Price Prediction API is running"
    )


def test_valid_prediction(client):
    # Read a real sample from the uploaded dataset.
    data = pd.read_csv(DATA_PATH)

    assert TARGET in data.columns, (
        f"Target column '{TARGET}' not found"
    )

    features = data.drop(columns=[TARGET])
    row = features.iloc[0]

    payload = {}

    for column, value in row.items():
        if pd.isna(value):
            payload[column] = None
        elif hasattr(value, "item"):
            payload[column] = value.item()
        else:
            payload[column] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, (
        f"Status: {response.status_code}; "
        f"Response: {response.get_data(as_text=True)}"
    )

    assert "predicted_sale_price" in response.json
    assert isinstance(
        response.json["predicted_sale_price"], (int, float)
    )


def test_missing_input(client):
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert "error" in response.json


def test_invalid_features(client):
    response = client.post(
        "/predict",
        json={"unknown_feature": 123}
    )

    assert response.status_code in (400, 500)
    assert "error" in response.json
