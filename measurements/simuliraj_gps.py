import csv
import math

# test to see if whole workflw works

csv_path = "logs/drive_20260920_064850.csv"

# starting point (NS)
start_lat = 45.25510
start_lon = 19.84510
altitude = 82.0

rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Movement simulation:
step_lat = 0.000028
step_lon = 0.000035

for i, row in enumerate(rows):
    cur_lat = start_lat + (i * step_lat) + (math.sin(i / 3.0) * 0.000010)
    cur_lon = start_lon + (i * step_lon) + (math.cos(i / 3.0) * 0.000010)

    row["lat"] = f"{cur_lat:.6f}"
    row["lon"] = f"{cur_lon:.6f}"
    row["altitude_m"] = f"{altitude + (i * 0.2):.1f}"
    row["speed_kmh"] = "4.3"
    row["course_deg"] = "48"
    row["hdop"] = "0.9"
    row["satellites"] = "11"
    row["gps_fix"] = "1"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully written GPS coordinates for {len(rows)} points to {csv_path}")
