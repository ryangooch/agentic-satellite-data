#!/usr/bin/env python
"""Download NEXRAD Level 2 data from AWS and optionally plot a radar variable."""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path

# The NEXRAD bucket is public — remove any AWS_PROFILE that would cause
# boto3 to fail looking for credentials we don't need.
os.environ.pop("AWS_PROFILE", None)

import nexradaws


def download_nexrad(radar: str, dt: datetime, output_dir: Path) -> Path:
    """Download the NEXRAD Level 2 scan closest to the given datetime.

    Args:
        radar: 4-letter NEXRAD radar ID (e.g. KTLX).
        dt: Target datetime (UTC).
        output_dir: Directory to save the downloaded file.

    Returns:
        Path to the downloaded file.
    """
    conn = nexradaws.NexradAwsInterface()
    scans = conn.get_avail_scans(dt.year, dt.month, dt.day, radar)
    if not scans:
        raise ValueError(
            f"No scans found for radar {radar} on {dt.strftime('%Y-%m-%d')}"
        )

    # Find scan closest to requested time
    target_ts = dt.timestamp()
    closest = min(scans, key=lambda s: abs(s.scan_time.timestamp() - target_ts))
    print(f"Downloading scan: {closest.filename} ({closest.scan_time} UTC)")

    results = conn.download(closest, str(output_dir))
    if results.failed:
        raise RuntimeError(f"Download failed: {results.failed}")

    downloaded = results.success[0]
    filepath = Path(output_dir) / downloaded.filename
    print(f"Saved to: {filepath}")
    return filepath


DUAL_POL_FIELDS = [
    "reflectivity",
    "differential_reflectivity",
    "cross_correlation_ratio",
    "specific_differential_phase",
]

DUAL_POL_LABELS = ["Z (dBZ)", "ZDR (dB)", "RhoHV", "KDP (deg/km)"]


def plot_radar(filepath: Path, output_png: Path | None = None):
    """Plot a 2x2 quadrant of dual-pol radar variables for hydrometeor classification."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyart

    radar = pyart.io.read_nexrad_archive(str(filepath))

    # Compute KDP from differential phase if not already present
    if "specific_differential_phase" not in radar.fields:
        kdp = pyart.retrieve.kdp_maesaka(radar)
        radar.add_field("specific_differential_phase", kdp[0])

    display = pyart.graph.RadarDisplay(radar)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(filepath.name, fontsize=14, fontweight="bold")

    for ax, field, label in zip(axes.flat, DUAL_POL_FIELDS, DUAL_POL_LABELS):
        plot_kwargs = {}
        if field == "differential_reflectivity":
            plot_kwargs = {"vmin": -4, "vmax": 4, "cmap": "RdBu_r"}
        elif field == "cross_correlation_ratio":
            plot_kwargs = {"vmin": 0.8, "vmax": 1.0}
        display.plot(field, 0, ax=ax, title=label, **plot_kwargs)
        display.plot_range_rings([50, 100, 150, 200], ax=ax, lw=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_png is None:
        output_png = filepath.with_suffix(".png")
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Download NEXRAD Level 2 data and optionally plot a radar variable."
    )
    parser.add_argument("radar", help="4-letter NEXRAD radar ID (e.g. KTLX)")
    parser.add_argument(
        "datetime",
        help="Target datetime in UTC (format: YYYY-MM-DDTHH:MM)",
        type=lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/nexrad",
        help="Output directory (default: data/nexrad)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a 2x2 quadrant of dual-pol variables (Z, ZDR, RhoHV, KDP) "
        "for hydrometeor classification",
    )
    parser.add_argument(
        "--plot-output",
        help="Path for the plot PNG (default: same name as data file with .png)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = download_nexrad(args.radar, args.datetime, output_dir)

    if args.plot:
        plot_output = Path(args.plot_output) if args.plot_output else None
        plot_radar(filepath, plot_output)


if __name__ == "__main__":
    main()
