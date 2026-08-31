from skyfield.api import load
import ssl
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

ts = load.timescale()
t = ts.now()

satellites = load.tle_file("https://celestrak.org/NORAD/elements/1999-025.txt")
print(f"Loaded {len(satellites)} satellites")
xs = []
for sat in satellites:
    geocentric = sat.at(t)
    pos = geocentric.position.km
    if not np.isnan(pos[0]):
        xs.append(pos[0])
print(f"Valid positions: {len(xs)}")
