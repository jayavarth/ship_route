import requests # type: ignore

def fetch_weather_data(api_key, latitude, longitude):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch weather data")

def fetch_bunker_cost():
    return {
        "price_per_ton": 450  
    }

def fetch_vessel_data(file_path='data/vessel_data.csv'):
    import pandas as pd
    return pd.read_csv(file_path)
