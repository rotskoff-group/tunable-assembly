import argparse

from fd import evaluate_solution

parser = argparse.ArgumentParser()
parser.add_argument("--d_iso", type=float, default=640)
parser.add_argument("--d_odd", type=float, default=640)
parser.add_argument("--flux_ratio", type=float, default=0.1)
parser.add_argument("--position", type=int, default=4)
parser.add_argument("--scheme", type=int, default=0)

args = parser.parse_args()
D_iso = args.d_iso
D_odd = args.d_odd
flux_ratio = args.flux_ratio
position = args.position
scheme = args.scheme

# Specify plate currents
flux = D_odd / D_iso * 1e1

evaluate_solution(D_iso, D_odd, flux, flux_ratio, position, scheme)
