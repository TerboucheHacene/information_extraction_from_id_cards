import requests

url = "http://localhost:5000/predict"

payload = {}
files = [
    (
        "image",
        (
            "325.jpeg",
            open(
                "/home/hacene/Documents/information_extraction_from_id_cards/data/raw/325.jpeg",
                "rb",
            ),
            "image/jpeg",
        ),
    )
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.json())
