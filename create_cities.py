import json
import re

# Comprehensive state base coordinates map
STATE_COORDS = {
    'AL': {'lat': 32.3182, 'lng': -86.9023},
    'AK': {'lat': 63.5887, 'lng': -154.4931},
    'AZ': {'lat': 34.0489, 'lng': -111.0937},
    'AR': {'lat': 34.9697, 'lng': -92.3731},
    'CA': {'lat': 36.7783, 'lng': -119.4179},
    'CO': {'lat': 39.5501, 'lng': -105.7821},
    'CT': {'lat': 41.6032, 'lng': -73.0877},
    'DE': {'lat': 38.9108, 'lng': -75.5277},
    'FL': {'lat': 27.6648, 'lng': -81.5158},
    'GA': {'lat': 32.1656, 'lng': -82.9001},
    'HI': {'lat': 19.8968, 'lng': -155.5828},
    'ID': {'lat': 44.0682, 'lng': -114.7420},
    'IL': {'lat': 40.6331, 'lng': -89.3985},
    'IN': {'lat': 39.8494, 'lng': -86.2583},
    'IA': {'lat': 41.8780, 'lng': -93.0977},
    'KS': {'lat': 38.5266, 'lng': -96.7265},
    'KY': {'lat': 37.8393, 'lng': -84.2700},
    'LA': {'lat': 31.2448, 'lng': -92.1450},
    'ME': {'lat': 45.2538, 'lng': -69.4455},
    'MD': {'lat': 39.0458, 'lng': -76.6413},
    'MA': {'lat': 42.4072, 'lng': -71.3824},
    'MI': {'lat': 44.3148, 'lng': -85.6024},
    'MN': {'lat': 46.7296, 'lng': -94.6859},
    'MS': {'lat': 32.3547, 'lng': -89.3985},
    'MO': {'lat': 37.9643, 'lng': -91.8318},
    'MT': {'lat': 46.8797, 'lng': -110.3626},
    'NE': {'lat': 41.4925, 'lng': -99.9018},
    'NV': {'lat': 38.8026, 'lng': -116.4194},
    'NH': {'lat': 43.1939, 'lng': -71.5724},
    'NJ': {'lat': 40.0583, 'lng': -74.4057},
    'NM': {'lat': 34.5199, 'lng': -105.8701},
    'NY': {'lat': 40.7128, 'lng': -74.0060},
    'NC': {'lat': 35.7596, 'lng': -79.0193},
    'ND': {'lat': 47.5515, 'lng': -101.0020},
    'OH': {'lat': 40.4173, 'lng': -82.9071},
    'OK': {'lat': 35.5653, 'lng': -96.9289},
    'OR': {'lat': 43.8041, 'lng': -120.5542},
    'PA': {'lat': 41.2033, 'lng': -77.1945},
    'RI': {'lat': 41.5801, 'lng': -71.4774},
    'SC': {'lat': 33.8361, 'lng': -81.1637},
    'SD': {'lat': 44.2998, 'lng': -99.4388},
    'TN': {'lat': 35.5175, 'lng': -86.5804},
    'TX': {'lat': 31.9686, 'lng': -99.9018},
    'UT': {'lat': 39.3210, 'lng': -111.0937},
    'VT': {'lat': 44.5588, 'lng': -72.5778},
    'VA': {'lat': 37.4316, 'lng': -78.6569},
    'WA': {'lat': 47.7511, 'lng': -120.7401},
    'WV': {'lat': 38.5976, 'lng': -80.4549},
    'WI': {'lat': 43.7844, 'lng': -88.7879},
    'WY': {'lat': 43.0759, 'lng': -107.2903},
    'DC': {'lat': 38.9072, 'lng': -77.0369}
}

MAJOR_CITIES = {
    'Chicago': {'lat': 41.8781, 'lng': -87.6298},
    'Los Angeles': {'lat': 34.0522, 'lng': -118.2437},
    'New York': {'lat': 40.7128, 'lng': -74.0060},
    'Miami': {'lat': 25.7617, 'lng': -80.1918},
    'Houston': {'lat': 29.7604, 'lng': -95.3698},
    'San Francisco': {'lat': 37.7749, 'lng': -122.4194},
    'Seattle': {'lat': 47.6062, 'lng': -122.3321},
    'Dayton': {'lat': 39.7589, 'lng': -84.1916},
    'Columbus': {'lat': 39.9612, 'lng': -82.9988},
    'Akron': {'lat': 41.0814, 'lng': -81.5190},
    'Cleveland': {'lat': 41.4993, 'lng': -81.6944},
    'Cincinnati': {'lat': 39.1031, 'lng': -84.5120},
    'Sacramento': {'lat': 38.5816, 'lng': -121.4944},
    'San Diego': {'lat': 32.7157, 'lng': -117.1611},
    'Dallas': {'lat': 32.7767, 'lng': -96.7970},
    'Austin': {'lat': 30.2672, 'lng': -97.7431},
    'Atlanta': {'lat': 33.7490, 'lng': -84.3880},
    'Denver': {'lat': 39.7392, 'lng': -104.9903},
    'Phoenix': {'lat': 33.4484, 'lng': -112.0740},
    'Las Vegas': {'lat': 36.1699, 'lng': -115.1398},
    'Portland': {'lat': 45.5152, 'lng': -122.6784},
    'Des Moines': {'lat': 41.5868, 'lng': -93.6250},
    'Omaha': {'lat': 41.2565, 'lng': -95.9345},
    'Detroit': {'lat': 42.3314, 'lng': -83.0458},
    'Philadelphia': {'lat': 39.9526, 'lng': -75.1652},
    'Pittsburgh': {'lat': 40.4406, 'lng': -79.9959},
    'Tehran': {'lat': 35.6892, 'lng': 51.3890}
}

def get_coords(city, state):
    if city in MAJOR_CITIES:
        return MAJOR_CITIES[city]['lat'], MAJOR_CITIES[city]['lng']
    st = STATE_COORDS.get(state, {'lat': 40.6331, 'lng': -89.3985})
    h = 0
    for ch in (city + state):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    lat_off = ((h % 200) - 100) / 500.0
    lng_off = (((h >> 3) % 200) - 100) / 500.0
    return round(st['lat'] + lat_off, 4), round(st['lng'] + lng_off, 4)

print("Script loaded")
