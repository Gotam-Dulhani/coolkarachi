"""Build the Karachi UHI feature stack in Google Earth Engine and export samples.

For each requested year this script:
  1. Builds a May-June Landsat 8/9 LST composite (30 m, deg C).
  2. Builds a May-June Sentinel-2 composite and derives NDVI/NDBI/NDWI/albedo.
  3. Adds Dynamic World land-cover fractions, SRTM elevation and WorldPop population.
  4. Samples 60,000 random points and exports a CSV to Google Drive.
  5. Exports the LST raster (GeoTIFF) for mapping.

Usage:
    python src/01_export_features.py --years 2020 2021 2022 2023 2024 2025
    python src/01_export_features.py --years 2024 --no-raster
"""

from __future__ import annotations

import argparse

import ee

from ee_setup import init

# Karachi Division bounding box + 15 km buffer (rural reference ring).
AOI_BBOX = [66.65, 24.70, 67.45, 25.15]
DRIVE_FOLDER = "karachi_uhi"
N_SAMPLES = 60_000
SCALE = 30
# WorldPop (100 m) is published up to 2020, so later years reuse the 2020 surface.
WORLDPOP_MAX_YEAR = 2020


def mask_landsat(img: ee.Image) -> ee.Image:
    """Mask clouds, cloud shadow, cirrus, snow using QA_PIXEL bits 1-5."""
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(0b111110).eq(0)
    return img.updateMask(clear)


def to_celsius(img: ee.Image) -> ee.Image:
    """Convert Landsat C2 L2 ST_B10 DN to degrees Celsius."""
    return (
        img.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST")
    )


def landsat_lst(year: int, region: ee.Geometry) -> ee.Image:
    start = f"{year}-05-01"
    end = f"{year}-06-30"
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUD_COVER", 40))
        .map(mask_landsat)
        .map(to_celsius)
    )
    return collection.median().rename("LST")


def mask_sentinel2(img: ee.Image) -> ee.Image:
    """Mask no-data, saturated, cloud shadow, cloud and cirrus using SCL."""
    scl = img.select("SCL")
    bad = (
        scl.eq(0)
        .Or(scl.eq(1))
        .Or(scl.eq(3))
        .Or(scl.eq(8))
        .Or(scl.eq(9))
        .Or(scl.eq(10))
    )
    return img.updateMask(bad.Not())


def sentinel2_indices(year: int, region: ee.Geometry) -> ee.Image:
    start = f"{year}-05-01"
    end = f"{year}-06-30"
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start, end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_sentinel2)
    )
    img = collection.median().multiply(0.0001)

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # Liang (2001) broadband albedo, adapted to Sentinel-2 bands (approximate).
    albedo = (
        img.select("B2").multiply(0.356)
        .add(img.select("B4").multiply(0.130))
        .add(img.select("B8").multiply(0.373))
        .add(img.select("B11").multiply(0.085))
        .add(img.select("B12").multiply(0.072))
        .subtract(0.0018)
        .rename("ALBEDO")
    )
    return ee.Image.cat([ndvi, ndbi, ndwi, albedo])


def dynamic_world(year: int, region: ee.Geometry) -> ee.Image:
    collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{year}-05-01", f"{year}-06-30")
        .filterBounds(region)
    )
    fractions = (
        collection.select(["built", "trees", "water"])
        .mean()
        .rename(["BUILT_FRAC", "TREE_FRAC", "WATER_FRAC"])
    )
    lulc = collection.select("label").mode().rename("LULC")
    return ee.Image.cat([fractions, lulc])


def worldpop(year: int, region: ee.Geometry) -> ee.Image:
    pop_year = min(year, WORLDPOP_MAX_YEAR)
    return (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", pop_year))
        .filterBounds(region)
        .mosaic()
        .select("population")
        .rename("POP")
    )


def build_stack(year: int, region: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("ELEV")
    lonlat = ee.Image.pixelLonLat().rename(["LON", "LAT"])
    return ee.Image.cat(
        [
            landsat_lst(year, region),
            sentinel2_indices(year, region),
            dynamic_world(year, region),
            dem,
            worldpop(year, region),
            lonlat,
        ]
    )


def export_year(
    year: int, region: ee.Geometry, export_raster: bool, urban_only: bool
) -> None:
    print(f"[{year}] building stack ...")
    stack = build_stack(year, region)

    if export_raster:
        lst = stack.select("LST")
        raster_task = ee.batch.Export.image.toDrive(
            image=lst,
            description=f"karachi_lst_{year}",
            folder=DRIVE_FOLDER,
            fileNamePrefix=f"karachi_lst_{year}",
            region=region,
            scale=SCALE,
            maxPixels=1e13,
        )
        raster_task.start()
        print(f"[{year}] raster export started -> {raster_task.id}")

    samples = None
    prefix = f"karachi_features_{year}"
    if urban_only:
        # Stratified sample on the built-up mask so we get N_SAMPLES urban
        # pixels regardless of how much of the AOI is rural.
        built = stack.select("BUILT_FRAC").gte(0.2).rename("URBAN")
        samples = stack.addBands(built).stratifiedSample(
            numPoints=0,
            classBand="URBAN",
            region=region,
            scale=SCALE,
            seed=42,
            classValues=[1],
            classPoints=[N_SAMPLES],
            dropNulls=True,
            tileScale=4,
            geometries=False,
        )
        prefix = f"karachi_ml_{year}"
    else:
        samples = stack.sample(
            region=region,
            scale=SCALE,
            numPixels=N_SAMPLES,
            geometries=False,
            seed=42,
            tileScale=4,
        )

    samples = samples.map(lambda feature: feature.set("YEAR", year))

    table_task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=prefix,
        folder=DRIVE_FOLDER,
        fileNamePrefix=prefix,
        fileFormat="CSV",
    )
    table_task.start()
    print(f"[{year}] table export started -> {table_task.id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024],
        help="Years to process (Landsat 8/9 available from 2013).",
    )
    parser.add_argument(
        "--no-raster",
        action="store_true",
        help="Skip GeoTIFF export and only export the sample CSV.",
    )
    parser.add_argument(
        "--urban-only",
        action="store_true",
        help="Sample only built-up pixels (Dynamic World built fraction >= 0.2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init()

    aoi = ee.Geometry.Rectangle(AOI_BBOX)
    region = aoi.buffer(15_000)  # rural reference ring for SUHI

    print(f"Region: {AOI_BBOX} + 15 km buffer")
    print(f"Years: {args.years}")

    for year in args.years:
        export_year(
            year,
            region,
            export_raster=not args.no_raster,
            urban_only=args.urban_only,
        )

    print("\nAll export tasks submitted.")
    print("Monitor: https://code.earthengine.google.com/tasks")
    print(f"Download from Google Drive folder: {DRIVE_FOLDER}")


if __name__ == "__main__":
    main()
