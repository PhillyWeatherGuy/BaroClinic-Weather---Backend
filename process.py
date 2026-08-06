import os
import datetime
import glob
import json
import math
import numpy as np
import xarray as xr
from PIL import Image
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
OUTPUT_DIST_DIR    = "run_conus"      
MAX_FORECAST_HOURS = 360              
FORECAST_STEPS     = [h for h in range(0, MAX_FORECAST_HOURS + 1) if h % 3 == 0]

TEMP_MIN_K = 210.0  # ~ -81.67°F
TEMP_MAX_K = 330.0  # ~  134.33°F

# Universally safe WebGL max texture size for desktop & mobile GPUs
MAX_TEXTURE_SIZE = 8192 


def process_grib_to_array(grib_path):
    """
    Reads a GRIB file, normalizes coordinates, clips to Web Mercator latitude 
    bounds (~85.0511° N/S), reprojects to EPSG:3857, and returns an 8-bit uint8 numpy array.
    """
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={'errors': 'ignore'})
    
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})

    # Ensure latitudes run North to South (+90 to -90)
    ds = ds.sortby('latitude', ascending=False)
    
    # Normalize longitudes to wrap cleanly from -180 to 180 degrees
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        ).sortby('longitude')

    # Clip latitudes to Web Mercator map limits
    ds = ds.sel(latitude=slice(85.051129, -85.051129))

    temp_data = ds['t2m'] if 't2m' in ds else ds[list(ds.data_vars)[0]]
    if temp_data.ndim > 2:
        temp_data = temp_data.squeeze()

    # Define CRS as WGS84 and reproject to Web Mercator (EPSG:3857)
    temp_data.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    temp_data.rio.write_crs("EPSG:4326", inplace=True)

    temp_mercator = temp_data.rio.reproject(
        "EPSG:3857",
        resampling=Resampling.cubic
    )

    arr = temp_mercator.values
    if arr.ndim > 2:
        arr = np.squeeze(arr)

    # Fill NaNs and normalize Kelvin temperatures to uint8 (0-255)
    arr = np.nan_to_num(arr, nan=TEMP_MIN_K)
    normalized = np.clip((arr - TEMP_MIN_K) / (TEMP_MAX_K - TEMP_MIN_K), 0.0, 1.0)
    gray_image = (normalized * 255.0).astype(np.uint8)

    ds.close()
    return gray_image


def build_spritesheet_chunks(frame_arrays, steps_written):
    """
    Packs in-memory 2D uint8 numpy arrays into 8192x8192 WebGL texture atlases.
    Outputs chunks ready for PNG export and manifest metadata.
    """
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    
    # Grid limits based on 8192x8192 WebGL max hardware texture bounds
    max_cols = max(1, MAX_TEXTURE_SIZE // frame_w)
    max_rows = max(1, MAX_TEXTURE_SIZE // frame_h)
    frames_per_sheet = max_cols * max_rows

    print(f"📊 Grid Math: Max {max_cols} cols x {max_rows} rows per sheet ({frame_w}x{frame_h} per frame).")
    print(f"📏 Max frames per spritesheet chunk: {frames_per_sheet}")

    chunks = []
    
    # Slice the total in-memory frames into safe-sized chunk arrays
    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_sheet)):
        chunk_frames = frame_arrays[i:i + frames_per_sheet]
        chunk_steps = steps_written[i:i + frames_per_sheet]
        
        sheet_w = frame_w * max_cols
        sheet_rows = math.ceil(len(chunk_frames) / max_cols)
        sheet_h = frame_h * sheet_rows
        
        spritesheet_arr = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

        for idx, arr in enumerate(chunk_frames):
            col = idx % max_cols
            row = idx // max_cols

            y_start = row * frame_h
            y_end = y_start + frame_h
            x_start = col * frame_w
            x_end = x_start + frame_w

            spritesheet_arr[y_start:y_end, x_start:x_end] = arr

        spritesheet_img = Image.fromarray(spritesheet_arr, mode='L')
        spritesheet_filename = f"t2m_spritesheet_{chunk_idx}.png"
        
        chunks.append({
            "image": spritesheet_img,
            "manifest_data": {
                "file": spritesheet_filename,
                "forecast_steps": chunk_steps,
                "columns": max_cols,
                "rows": sheet_rows,
                "sheet_width": sheet_w,
                "sheet_height": sheet_h
            }
        })

    return chunks, frame_w, frame_h


def upload_to_b2(folder_path, bucket_name="baroclinic-weather-data"):
    """
    Uploads all generated files in the output directory to Backblaze B2 using 
    credentials supplied by environment variables (GitHub Secrets).
    """
    endpoint = os.environ.get("B2_ENDPOINT")
    key_id = os.environ.get("B2_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY")

    if not all([endpoint, key_id, app_key]):
        print("⚠️ B2 Credentials not set in environment. Skipping cloud upload.")
        return

    print("\n☁️ Uploading generated assets to Backblaze B2...")

    # Initialize S3 Client targeting Backblaze
    s3_client = boto3.client(
        service_name='s3',
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key
    )

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            # Determine content type (JSON vs PNG)
            content_type = "application/json" if filename.endswith(".json") else "image/png"
            
            try:
                s3_client.upload_file(
                    filepath,
                    bucket_name,
                    filename,
                    ExtraArgs={'ContentType': content_type}
                )
                print(f"  ✅ Uploaded to B2: {filename}")
            except Exception as e:
                print(f"  ❌ Failed to upload {filename}: {e}")


def run_master_pipeline():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_hour = now_utc.hour

    if current_hour >= 20:
        CHOSEN_RUN, target_date = "12", now_utc.strftime("%Y%m%d")
    elif current_hour >= 14:
        CHOSEN_RUN, target_date = "06", now_utc.strftime("%Y%m%d")
    elif current_hour >= 8:
        CHOSEN_RUN, target_date = "00", now_utc.strftime("%Y%m%d")
    elif current_hour >= 2:
        CHOSEN_RUN = "18"
        target_date = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")
    else:
        CHOSEN_RUN = "12"
        target_date = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d")

    print(f"🌍 UTC: {current_hour:02d}z → Selected ECMWF run: {CHOSEN_RUN}z on {target_date}")

    # Remove stale GRIB files from previous interrupted runs
    for f in glob.glob("ecmwf_t2m_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    client = Client(source="azure", model="ifs", resol="0p25")
    os.makedirs(OUTPUT_DIST_DIR, exist_ok=True)

    frame_arrays = []
    steps_written = []

    for step in FORECAST_STEPS:
        print(f"\n⏰ Downloading & processing F{step:03d}...")
        grib_file = f"ecmwf_t2m_{step:03d}.grib2"

        try:
            client.retrieve(
                date=target_date, time=int(CHOSEN_RUN), step=step,
                type="fc", levtype="sfc", param=["2t"], target=grib_file
            )
        except Exception as e:
            print(f"  ❌ Download failed for F{step:03d}: {e}")
            continue

        if os.path.exists(grib_file):
            try:
                # Process GRIB into memory array
                frame_arr = process_grib_to_array(grib_file)
                frame_arrays.append(frame_arr)
                steps_written.append(step)
                print(f"  ⚡ Processed frame array for F{step:03d} into memory buffer")
            except Exception as err:
                print(f"  ⚠️ Error generating array F{step:03d}: {err}")

            # Immediately delete raw GRIB file
            try: os.remove(grib_file)
            except Exception: pass

    if not frame_arrays:
        print("❌ No frames were processed. Exiting pipeline.")
        return

    # Pass in-memory frame arrays into the chunking engine
    chunks, frame_w, frame_h = build_spritesheet_chunks(frame_arrays, steps_written)

    manifest_chunks = []
    
    # Save ONLY the combined spritesheet PNG chunks to disk
    for chunk in chunks:
        filename = chunk["manifest_data"]["file"]
        filepath = os.path.join(OUTPUT_DIST_DIR, filename)
        
        chunk["image"].save(filepath, format='PNG', optimize=True)
        manifest_chunks.append(chunk["manifest_data"])
        print(f"  💾 Saved spritesheet: {filename}")

    # Build master manifest JSON
    manifest = {
        "model": "ecmwf",
        "parameter": "2t",
        "type": "spritesheet_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": TEMP_MIN_K,
        "temp_max_k": TEMP_MAX_K,
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    manifest_path = os.path.join(OUTPUT_DIST_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n🎉 Pipeline Finished! {len(chunks)} spritesheet(s) and manifest ready in {OUTPUT_DIST_DIR}/")

    # Upload outputs directly to Backblaze B2
    upload_to_b2(OUTPUT_DIST_DIR)


if __name__ == "__main__":
    run_master_pipeline()