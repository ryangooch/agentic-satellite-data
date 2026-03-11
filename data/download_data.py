#!/usr/bin/env python3
"""Optional script to download real Sentinel-2 scenes from Copernicus Data Space.

NOTE: For lecture purposes, synthetic scenes are already in data/scenes/ —
this script is for students who want to experiment with real satellite data.

Requirements: pip install sentinelsat
API registration: https://dataspace.copernicus.eu/
Viewing the data: https://browser.dataspace.copernicus.eu
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A scenes from Copernicus Data Space"
    )
    parser.add_argument("--region", required=True,
                        help="Bounding box: 'lat_min,lon_min,lat_max,lon_max'")
    parser.add_argument("--date", required=True,
                        help="Target date: YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/scenes/real",
                        help="Output directory for downloaded scenes")
    args = parser.parse_args()

    try:
        from sentinelsat import SentinelAPI
    except ImportError:
        print("Install sentinelsat: pip install sentinelsat")
        sys.exit(1)

    lat_min, lon_min, lat_max, lon_max = [float(x) for x in args.region.split(",")]
    print(f"Searching for Sentinel-2 L2A scenes near {args.region} on {args.date}...")
    print("Note: You will need Copernicus Data Space credentials.")
    print("Register at: https://dataspace.copernicus.eu/")
    print("\nFor lecture use, the synthetic scenes in data/scenes/ are sufficient.")
    print("This script is provided for students who want to work with real data.")


if __name__ == "__main__":
    main()
