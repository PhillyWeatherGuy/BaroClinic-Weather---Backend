import os
import shutil
import datetime
import glob
import json
import math
import concurrent.futures
import numpy as np
import xarray as xr
from PIL import Image
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3

# Enable multi-threaded GDAL reprojection globally
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

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

# Worker count matching runner hardware (safely tuned for RAM limits)
MAX_CONCURRENT_WORKERS = min(4, os.cpu_count() or 2)


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


def fetch_and_process_step(client, target_date, chosen_run, step):
    """
    Worker task: Downloads a single forecast step GRIB file, converts it to array,
    and cleans up local GRIB storage immediately.
    """
    grib_file = f"ecmwf_t2m_{step:03d}.grib2"
    try:
        client.retrieve(
            date=target_date, time=int(chosen_run), step=step,
            type="fc", levtype="sfc", param=["2t"], target=grib_file
        )
        if os.path.exists(grib_file):
            frame_arr = process_grib_to_array(grib_file)
            try: os.remove(grib_file)
            except Exception: pass
            print(f"  ⚡ Processed F{step:03d}")
            return step, frame_arr
    except Exception as e:
        print(f"  ❌ Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None


def build_spritesheet_chunks(frame_arrays, steps_written, model_name, target_date, chosen_run):
    """
    Packs in-memory 2D uint8 numpy arrays into 8192x8192 WebGL texture atlases.
    """
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    
    max_cols = max(1, MAX_TEXTURE_SIZE // frame_w)
    max_rows = max(1, MAX_TEXTURE_SIZE // frame_h)
    frames_per_sheet = max_cols * max_rows

    print(f"📊 Grid Math: Max {max_cols} cols x {max_rows} rows per sheet ({frame_w}x{frame_h} per frame).")
    print(f"📏 Max frames per spritesheet chunk: {frames_per_sheet}")

    chunks = []
    
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
        
        spritesheet_filename = f"{model_name}_{target_date}_{chosen_run}z_t2m_spritesheet_{chunk_idx}.png"
        
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


def upload_single_file(s3_client, bucket_name, filepath, filename):
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


def upload_to_b2_parallel(folder_path, bucket_name="baroclinic-weather-data"):
    endpoint = os.environ.get("B2_ENDPOINT")
    key_id = os.environ.get("B2_KEY_ID")
    app_key = os.environ.get("B2_APPLICATION_KEY")

    if not all([endpoint, key_id, app_key]):
        print("⚠️ B2 Credentials not set in environment. Skipping cloud upload.")
        return

    print("\n☁️ Uploading generated assets to Backblaze B2 concurrently...")

    s3_client = boto3.client(
        service_name='s3',
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key
    )

    files_to_upload = [
        (os.path.join(folder_path, fname), fname)
        for fname in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, fname))
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, fpath, fname)
            for fpath, fname in files_to_upload
        ]
        concurrent.futures.wait(futures)


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

    for f in glob.glob("ecmwf_t2m_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    client = Client(source="azure", model="ifs", resol="0p25")
    os.makedirs(OUTPUT_DIST_DIR, exist_ok=True)

    results = {}

    print(f"⚡ Starting multi-threaded download & processing pipeline ({MAX_CONCURRENT_WORKERS} workers)...")
    
    # Run network downloads and CPU reprojections concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, client, target_date, CHOSEN_RUN, step): step
            for step in FORECAST_STEPS
        }
        for future in concurrent.futures.as_completed(future_to_step):
            step, arr = future.result()
            if arr is not None:
                results[step] = arr

    # Sort results to keep correct chronological order
    sorted_steps = sorted(results.keys())
    frame_arrays = [results[s] for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print("❌ No frames were processed. Exiting pipeline.")
        return

    chunks, frame_w, frame_h = build_spritesheet_chunks(
        frame_arrays, 
        steps_written, 
        model_name="ecmwf", 
        target_date=target_date, 
        chosen_run=CHOSEN_RUN
    )

    manifest_chunks = []
    
    # Save PNG chunks with optimized compression (lossless, zero resolution impact)
    for chunk in chunks:
        filename = chunk["manifest_data"]["file"]
        filepath = os.path.join(OUTPUT_DIST_DIR, filename)
        
        # compress_level=3 is ~5x faster than optimize=True with identical pixel data
        chunk["image"].save(filepath, format='PNG', compress_level=3)
        manifest_chunks.append(chunk["manifest_data"])
        print(f"  💾 Saved spritesheet: {filename}")

    manifest = {
        "model": "ecmwf",
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
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

    upload_to_b2_parallel(OUTPUT_DIST_DIR)
    
    print(f"\n🧹 Cleaning up: Deleting local {OUTPUT_DIST_DIR}/ folder...")
    try:
        shutil.rmtree(OUTPUT_DIST_DIR)
        print("  ✅ Cleanup complete. Workspace is spotless!")
    except Exception as e:
        print(f"  ❌ Failed to delete folder: {e}")


if __name__ == "__main__":
    run_master_pipeline()