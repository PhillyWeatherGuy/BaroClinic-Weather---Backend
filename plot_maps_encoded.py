import os
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from scipy.ndimage import gaussian_filter

# ==============================================================================
# ENCODING CONSTANTS — NATIVE ECMWF 0.25° RESOLUTION
# ==============================================================================
FRAME_W   = 1440   # px — Native ECMWF 0.25° longitude grid (360° / 0.25°)
FRAME_H   = 721    # px — Native ECMWF 0.25° latitude grid (180° / 0.25° + 1)

GLOBAL_EXTENT = [-180.0, 180.0, -90.0, 90.0]

# Temperature encoding ranges (Fahrenheit → Kelvin)
TEMP_MIN_F = -70.0
TEMP_MAX_F = 130.0
TEMP_MIN_K = 273.15 + (TEMP_MIN_F - 32) * 5 / 9   # 216.483 K
TEMP_MAX_K = 273.15 + (TEMP_MAX_F - 32) * 5 / 9   # 316.483 K
TEMP_RANGE_K = TEMP_MAX_K - TEMP_MIN_K


# ==============================================================================
# CORE ENCODING FUNCTION
# ==============================================================================

def encode_temperature_to_grayscale(temp_kelvin: np.ndarray) -> np.ndarray:
    """
    Normalizes temperature (Kelvin) to 0-255 uint8 for raster tile encoding.
    Values outside [TEMP_MIN_K, TEMP_MAX_K] are safely clipped at boundaries.
    """
    clipped = np.clip(temp_kelvin, TEMP_MIN_K, TEMP_MAX_K)
    normalized = (clipped - TEMP_MIN_K) / TEMP_RANGE_K
    encoded = (normalized * 255.0).astype(np.uint8)
    return encoded


# ==============================================================================
# MAIN RENDER FUNCTION (GEOTIFF OUTPUT)
# ==============================================================================

def render_global_frame_encoded(grib_path: str, step: int, output_dir: str) -> str | None:
    """
    Reads ECMWF GRIB2, normalizes temperature to 0-255 uint8 grayscale,
    and exports a spatially-aware GeoTIFF (EPSG:4326) for tile processing.
    """
    if not os.path.exists(grib_path):
        print(f"  ⚠️  GRIB file not found: {grib_path}")
        return None

    try:
        with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={'errors': 'ignore'}) as ds:
            if 'lon' in ds.coords:
                ds = ds.rename({'lon': 'longitude'})
            if 'lat' in ds.coords:
                ds = ds.rename({'lat': 'latitude'})
            
            # Sort latitude DESCENDING (North +90° to South -90°)
            ds = ds.sortby('latitude', ascending=False)
            
            # Standardize longitude coordinates to [-180, 180]
            if ds.longitude.max() > 180:
                ds = ds.assign_coords(
                    longitude=(((ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')

            t2m_raw = ds['t2m'].load().squeeze()

        # Gaussian smoothing (0.65 sigma for global scale)
        t2m_k = gaussian_filter(t2m_raw.values, sigma=0.65)

    except Exception as e:
        print(f"  ⚠️  GRIB read failed for step F{step:03d}: {e}")
        return None

    # Encode array to 0-255 uint8 grayscale
    encoded_data = encode_temperature_to_grayscale(t2m_k)

    # Re-attach spatial dimensions & coordinates using xarray
    encoded_da = xr.DataArray(
        encoded_data,
        coords=[t2m_raw.latitude, t2m_raw.longitude],
        dims=["y", "x"]
    )

    # Assign WGS 84 spatial reference system
    encoded_da = encoded_da.rio.write_crs("EPSG:4326")

    # Output GeoTIFF filename
    out_filename = os.path.join(output_dir, f"t2m_encoded_F{step:03d}.tif")
    
    # Save as lossless LZW-compressed single-band uint8 GeoTIFF
    encoded_da.rio.to_raster(
        out_filename,
        dtype=rasterio.uint8,
        compress='lzw'
    )

    return out_filename