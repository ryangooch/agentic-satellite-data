# Sentinel-2 Spectral Indices for Agricultural Monitoring
## Quick Reference Guide

### Sentinel-2 Band Specifications

| Band | Name | Wavelength (nm) | Resolution | Primary Use |
|------|------|-----------------|------------|-------------|
| B02 | Blue | 490 | 10m | Atmospheric correction, EVI |
| B03 | Green | 560 | 10m | NDWI, visual interpretation |
| B04 | Red | 665 | 10m | NDVI, chlorophyll absorption |
| B08 | NIR | 842 | 10m | NDVI, vegetation structure |
| B8A | Red Edge | 865 | 20m | Vegetation stress detection |
| B11 | SWIR-1 | 1610 | 20m | NDWI, soil/vegetation moisture |
| B12 | SWIR-2 | 2190 | 20m | Soil moisture, mineral mapping |

### NDVI — Normalized Difference Vegetation Index

**Formula**: (B08 - B04) / (B08 + B04)

**Interpretation for Central Valley agriculture**:
- 0.80 - 1.00: Very dense vegetation (irrigated corn at peak, dense alfalfa)
- 0.60 - 0.80: Healthy orchard canopy (almonds, walnuts, citrus)
- 0.40 - 0.60: Moderate vegetation (young orchards, row crops mid-season)
- 0.20 - 0.40: Sparse vegetation or stressed crops
- 0.00 - 0.20: Bare soil, harvested fields, fallowed land
- < 0.00: Water bodies, impervious surfaces

**Limitations**: Saturates in very dense canopy (NDVI > 0.8 all looks the same). Use EVI instead.

### NDWI — Normalized Difference Water Index

**Formula**: (B03 - B11) / (B03 + B11)

**Interpretation**:
- > 0.3: Open water
- 0.0 to 0.3: Wet vegetation, saturated soil
- -0.1 to 0.0: Moderate moisture (normal irrigated crop)
- < -0.1: Dry vegetation, water-stressed crop

**Key application**: Distinguishes water stress from other stress types. A crop with low NDVI
but normal NDWI is likely suffering from nutrient deficiency, pest damage, or disease —
not drought.

### EVI — Enhanced Vegetation Index

**Formula**: 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)

**Advantages over NDVI**:
- Less saturation in dense canopy
- Reduced atmospheric effects (blue band correction)
- Better signal in high-biomass regions

**Typical ranges for Central Valley**:
- Healthy irrigated crops: 0.3 - 0.6
- Stressed crops: 0.15 - 0.3
- Bare soil: 0.0 - 0.15

### Multi-Index Analysis Strategy

For reliable crop health assessment, use multiple indices together:

1. **Start with NDVI** — broadest indicator of vegetation presence and health
2. **Cross-check with EVI** — if NDVI saturates (> 0.8), EVI provides better discrimination
3. **Add NDWI** — determines if stress is water-related or other cause
4. **Temporal comparison** — single-date analysis can be misleading; compare to:
   - Same field, 2-4 weeks ago (recent change detection)
   - Same field, same time last year (seasonal anomaly detection)
   - Neighboring fields, same date (local benchmarking)

### Common Pitfalls

1. **Cloud shadows**: Can cause false stress signals. Check the SCL (Scene Classification Layer)
   band for cloud shadow pixels (class 3).
2. **Mixed pixels**: At 10m resolution, field edges mix crop with roads, buildings, or bare soil.
   Buffer inward by 1-2 pixels when computing field-level statistics.
3. **Phenology effects**: A corn field after harvest (NDVI < 0.2) is not stressed — it's harvested.
   Always consider the crop calendar.
4. **Irrigation timing**: NDWI can fluctuate by 0.05-0.10 depending on when irrigation occurred
   relative to the satellite overpass. Sentinel-2 overpasses the Central Valley around 10:30 AM local time.
