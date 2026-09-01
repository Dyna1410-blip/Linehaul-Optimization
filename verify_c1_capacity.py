"""
Run this from the project root:  python3 verify_c1_capacity.py

Directly checks C1 (vehicle capacity, phy AND vol) on the REAL
vehicle_routes.csv — only ever verified on a tiny synthetic test before.
If pack_orders_into_vehicles has any systematic over-packing bug, THIS
would explain artificially cheap costs (fewer vehicles than physically
required to carry the real weight/volume).
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
cap_lookup = {r.vehicle_type: (r.phy_cap_kg, r.vol_cap_kg) for r in tables['vehicles'].itertuples(index=False)}

routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])

phy_violations = []
vol_violations = []
for row in routes.itertuples(index=False):
    caps = cap_lookup.get(row.vehicle_type)
    if caps is None:
        continue
    phy_cap, vol_cap = caps
    if row.phy_load_kg > phy_cap + 1e-6:
        phy_violations.append({'vehicle_id': row.vehicle_id, 'vehicle_type': row.vehicle_type,
                                'phy_load_kg': row.phy_load_kg, 'phy_cap_kg': phy_cap,
                                'overage_pct': (row.phy_load_kg / phy_cap - 1) * 100})
    if row.vol_load_kg > vol_cap + 1e-6:
        vol_violations.append({'vehicle_id': row.vehicle_id, 'vehicle_type': row.vehicle_type,
                                'vol_load_kg': row.vol_load_kg, 'vol_cap_kg': vol_cap,
                                'overage_pct': (row.vol_load_kg / vol_cap - 1) * 100})

print(f"Total vehicle dispatches checked: {len(routes)}")
print(f"PHYSICAL weight violations: {len(phy_violations)}")
print(f"VOLUMETRIC weight violations: {len(vol_violations)}")

if phy_violations:
    print("\nWorst physical weight violations:")
    print(pd.DataFrame(phy_violations).sort_values('overage_pct', ascending=False).head(10).to_string(index=False))
if vol_violations:
    print("\nWorst volumetric weight violations:")
    print(pd.DataFrame(vol_violations).sort_values('overage_pct', ascending=False).head(10).to_string(index=False))

# also check: does average utilization look suspiciously high (near 100%
# on almost every dispatch), which would hint at silent over-optimistic packing?
routes['phy_cap'] = routes['vehicle_type'].map(lambda v: cap_lookup.get(v, (None, None))[0])
routes['phy_util'] = routes['phy_load_kg'] / routes['phy_cap']
print(f"\nPhysical utilization stats across all dispatches:")
print(routes['phy_util'].describe())
