"""Quick connectivity check for Earth Engine.

Run: python src/00_check.py
"""

import ee

from ee_setup import PROJECT, init


def main() -> None:
    try:
        init()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Earth Engine init failed.\n"
            "Run:  earthengine authenticate\n"
            "Then retry. Original error:\n"
            f"{exc}"
        ) from exc

    print(f"Project: {PROJECT}")
    print("Arithmetic check:", ee.Number(1).add(2).getInfo())

    karachi = ee.Geometry.Point([67.0011, 24.8607])

    lst = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate("2024-05-01", "2024-06-30")
        .filterBounds(karachi)
        .filter(ee.Filter.lt("CLOUD_COVER", 40))
        .first()
        .select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST")
    )

    stats = lst.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=karachi.buffer(5000),
        scale=30,
        maxPixels=1e9,
    ).getInfo()

    print("Mean LST near Karachi (C):", stats)
    print("Earth Engine OK")


if __name__ == "__main__":
    main()
