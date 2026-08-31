from skyfield.api import load, EarthSatellite
import json

ts = load.timescale()
with open("celestrak_active.json", "r", encoding="utf-8") as f:
    omm_data = json.load(f)

print(f"Loaded {len(omm_data)} objects")
sat = EarthSatellite.from_omm(ts, omm_data[0])
print("Successfully parsed first object:", sat.name)
t = ts.now()
print("Position:", sat.at(t).position.km)
