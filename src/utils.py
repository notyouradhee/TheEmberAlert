import pandas as pd

MAP_KEY = "0c350a4e033ce53c6b9f43a88cfe9a71"  # Replace with your real key
url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}"

try:
    status = pd.read_json(url, typ='series')
    print(status)
except Exception as e:
    print("Error checking MAP_KEY status:", e)
