import os
import shutil
import datetime
import glob
import json
import math
import concurrent.futures
import numpy as np
import xarray as xr
import cv2
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3

# 🌟 Thread-Safe Isolated Matplotlib Figure Engine
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Enable multi-threaded GDAL reprojection globally
os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
OUTPUT_DIST_DIR    = "run_conus"      
MAX_FORECAST_HOURS = 360              
FORECAST_STEPS     = [h for h in range(0, MAX_FORECAST_HOURS + 1) if h % 3 == 0]

# 🌟 Exact Kelvin bounds matching Colab scale (-70.0°F to +130.0°F)
TEMP_MIN_K = 216.4833  # Exact -70.0°F
TEMP_MAX_K = 327.5944  # Exact +130.0°F

# Universally safe WebGL max texture size for desktop & mobile GPUs
MAX_TEXTURE_SIZE = 4096 

# Increased to 8 to maximize network pipeline throughput
MAX_CONCURRENT_WORKERS = 8


def extract_contour_geojson(raw_arr_k, target_k=273.15):
    """
    🌟 Global Cyclic Sub-Pixel Contour Extractor
    Uses cyclic column wrapping to seamlessly cross the 180° Anti-Meridian
    without drawing vertical Date-Line border seams!
    """
    try:
        frame_h, frame_w = raw_arr_k.shape
        
        # 1. OpenCV C++ Gaussian Blur to smooth grid steps
        smoothed = cv2.GaussianBlur(raw_arr_k.astype(np.float32), (5, 5), 1.2)
        smoothed_flipped = np.flipud(smoothed)
        
        # 2. 🌟 Add 1 cyclic column to wrap longitudes smoothly from -180° to +180.25°
        smoothed_cyclic = np.hstack([smoothed_flipped, smoothed_flipped[:, :1]])
        
        lon_step = 360.0 / frame_w
        lons = np.linspace(-180.0, 180.0 + lon_step, frame_w + 1)
        lats = np.linspace(-90.0, 90.0, frame_h)  # Strictly increasing!
            
        # 3. Extract sub-pixel floating-point isolines
        fig = Figure(figsize=(1, 1))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        cs = ax.contour(lons, lats, smoothed_cyclic, levels=[target_k])
        
        segments = []
        
        if hasattr(cs, 'allsegments') and len(cs.allsegments) > 0:
            level_lines = cs.allsegments[0]
            for line_array in level_lines:
                if len(line_array) >= 2:
                    pts = []
                    for pt in line_array:
                        lng = float(pt[0])
                        lat = float(pt[1])
                        if lng > 180.0:
                            lng = 180.0
                        pts.append([round(lng, 4), round(lat, 4)])
                    
                    # 🌟 Filter out fake border lines sitting on exact -180.0 / +180.0 margin
                    all_on_left = all(abs(p[0] - (-180.0)) < 0.01 for p in pts)
                    all_on_right = all(abs(p[0] - 180.0) < 0.01 for p in pts)
                    
                    if not all_on_left and not all_on_right:
                        segments.append(pts)
                            
        fig.clear()

        if not segments:
            print("  ⚠️ Note: 0 contour segments generated.")
            return {"type": "FeatureCollection", "features": []}

        print(f"  ✨ Generated {len(segments)} smooth 32°F contour segments")
        return {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": segments},
                "properties": {
                    "name": "32°F Freezing Line",
                    "color": "#4169E1",
                    "width": 2.0,
                    "opacity": 0.95
                }
            }]
        }
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return {"type": "FeatureCollection", "features": []}


def process_grib_to_array(grib_path, parameter):
    """
    Reads a GRIB file, normalizes coordinates, retains full 90°N to -90°S latitude 
    coverage (EPSG:4326 Equirectangular), extracts vector contours, and scales uint8 array.
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

    # Select dynamic parameter variable safely
    target_var = parameter if parameter in ds else list(ds.data_vars)[0]
    data_array = ds[target_var]

    # Extract raw numpy array in Kelvin (32-bit float precision)
    raw_arr_k = np.squeeze(data_array.values)
    ds.close()

    # 🌟 Extract sub-pixel smooth 32°F Vector Contour JSON safely
    contour_geojson = extract_contour_geojson(raw_arr_k, target_k=273.15)

    # In-place uint8 memory normalization for WebGL texture atlas
    arr = np.nan_to_num(raw_arr_k, copy=False, nan=TEMP_MIN_K)
    np.clip(arr, TEMP_MIN_K, TEMP_MAX_K, out=arr)
    arr -= TEMP_MIN_K
    arr /= (TEMP_MAX_K - TEMP_MIN_K)
    arr *= 255.0

    return arr.astype(np.uint8), contour_geojson


def fetch_and_process_step(client, target_date, chosen_run, step, parameter, model_name):
    """
    Worker task: Downloads a single forecast step GRIB file, converts it to array + contour JSON,
    and cleans up local GRIB storage immediately.
    """
    grib_file = f"{model_name}_{parameter}_{step:03d}.grib2"
    try:
        client.retrieve(
            date=target_date, time=int(chosen_run), step=step,
            type="fc", levtype="sfc", param=[parameter], target=grib_file
        )
        if os.path.exists(grib_file):
            frame_arr, contour_geojson = process_grib_to_array(grib_file, parameter)
            try: os.remove(grib_file)
            except Exception: pass
            print(f"  ⚡ Processed F{step:03d}")
            return step, frame_arr, contour_geojson
    except Exception as e:
        print(f"  ❌ Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None, None


def build_spritesheet_chunks(frame_arrays, steps_written, model_name, parameter, target_date, chosen_run):
    """
    Packs in-memory 2D uint8 numpy arrays into 4096x4096 WebGL texture atlases.
    """
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    
    max_cols = max(1, MAX_TEXTURE_SIZE // frame_w)
    max_rows = max(1, MAX_TEXTURE_SIZE // frame_h)
    frames_per_sheet = max_cols * max_rows

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

        spritesheet_filename = f"{model_name}_{parameter}_{target_date}_{chosen_run}z_spritesheet_{chunk_idx}.png"
        
        chunks.append({
            "array": spritesheet_arr,
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, fpath, fname)
            for fpath, fname in files_to_upload
        ]
        concurrent.futures.wait(futures)


def run_master_pipeline():
    MODEL_NAME = "ecmwf"
    PARAMETER = "2t"

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

    init_time_iso = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}T{CHOSEN_RUN}:00:00Z"

    print(f"🌍 Model: {MODEL_NAME} | Param: {PARAMETER} | Run: {CHOSEN_RUN}z on {target_date}")

    for f in glob.glob(f"{MODEL_NAME}_{PARAMETER}_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    client = Client(source="azure", model="ifs", resol="0p25")
    os.makedirs(OUTPUT_DIST_DIR, exist_ok=True)

    results = {}
    contours_dict = {}

    print(f"⚡ Starting multi-threaded pipeline ({MAX_CONCURRENT_WORKERS} workers)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, client, target_date, CHOSEN_RUN, step, PARAMETER, MODEL_NAME): step
            for step in FORECAST_STEPS
        }
        for future in concurrent.futures.as_completed(future_to_step):
            step, arr, contour_json = future.result()
            if arr is not None:
                results[step] = arr
                contours_dict[step] = contour_json

    sorted_steps = sorted(results.keys())
    frame_arrays = [results[s] for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print("❌ No frames were processed. Exiting pipeline.")
        return

    # 🌟 Save static vector contour JSON files per forecast step
    for step in sorted_steps:
        if step in contours_dict and contours_dict[step]:
            contour_filename = f"{MODEL_NAME}_{PARAMETER}_{target_date}_{CHOSEN_RUN.lower()}z_f{step:03d}_contours.json"
            c_path = os.path.join(OUTPUT_DIST_DIR, contour_filename)
            with open(c_path, 'w') as f:
                json.dump(contours_dict[step], f)
            
            # Also save latest static reference copy
            latest_c_path = os.path.join(OUTPUT_DIST_DIR, f"{MODEL_NAME}_{PARAMETER}_f{step:03d}_contours.json")
            with open(latest_c_path, 'w') as f:
                json.dump(contours_dict[step], f)

    chunks, frame_w, frame_h = build_spritesheet_chunks(
        frame_arrays, 
        steps_written, 
        model_name=MODEL_NAME, 
        parameter=PARAMETER,
        target_date=target_date, 
        chosen_run=CHOSEN_RUN
    )

    manifest_chunks = []
    
    for chunk in chunks:
        filename = chunk["manifest_data"]["file"]
        filepath = os.path.join(OUTPUT_DIST_DIR, filename)
        
        cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        manifest_chunks.append(chunk["manifest_data"])
        print(f"  💾 Saved spritesheet: {filename}")

    manifest = {
        "model": MODEL_NAME,
        "parameter": PARAMETER,
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "spritesheet_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": TEMP_MIN_K,
        "temp_max_k": TEMP_MAX_K,
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    run_manifest_filename = f"{MODEL_NAME}_{target_date}_{CHOSEN_RUN.lower()}z_manifest.json"
    for m_fname in ["manifest.json", run_manifest_filename]:
        m_path = os.path.join(OUTPUT_DIST_DIR, m_fname)
        with open(m_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  📄 Saved manifest: {m_fname}")

    print(f"\n🎉 Pipeline Finished! Spritesheets, JSON contours, and manifests ready in {OUTPUT_DIST_DIR}/")

    upload_to_b2_parallel(OUTPUT_DIST_DIR)
    
    print(f"\n🧹 Cleaning up: Deleting local {OUTPUT_DIST_DIR}/ folder...")
    try:
        shutil.rmtree(OUTPUT_DIST_DIR)
        print("  ✅ Cleanup complete. Workspace is spotless!")
    except Exception as e:
        print(f"  ❌ Failed to delete folder: {e}")


if __name__ == "__main__":
    run_master_pipeline()