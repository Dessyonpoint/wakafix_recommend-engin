import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "service_type": "cleaner",
    "location": "Dawaki",
    "time_slot": "12:00PM",
    "top_k": 3
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())