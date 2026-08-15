import os
import shutil
import datetime
import glob
import json
import math
import sys
import concurrent.futures
import numpy as np
import xarray as xr
import cv2
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3
import contourpy
import gzip  # 🌟 Added gzip for compressed binary output

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

MAX_FORECAST_HOURS = 360
FORECAST_STEPS = [h for h in range(0, MAX_FORECAST_HOURS + 1) if h % 3 == 0]

MAX_TEXTURE_SIZE = 4096

MAX_CONCURRENT_WORKERS = 4

MAX_CONCURRENT_PARAMS = 2

CONFIG_FILE_PATH = os.path.join("config", "parameters.json")


def load_parameter_config(param_key="2t"):
    """
    Loads parameter configuration from config/parameters.json
    """
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            params = data.get("parameters", {})
            if param_key in params:
                return params[param_key]
    raise FileNotFoundError(f"Parameter '{param_key}' not found in {CONFIG_FILE_PATH}")


def split_path_at_dateline(vertices, max_jump=180.0):
    if len(vertices) < 2:
        return []

    split_paths = []
    current_path = [vertices[0]]

    for i in range(1, len(vertices)):
        prev_pt = vertices[i - 1]
        curr_pt = vertices[i]

        if abs(curr_pt[0] - prev_pt[0]) > max_jump:
            if len(current_path) >= 2:
                split_paths.append(current_path)
            current_path = [curr_pt]
        else:
            current_path.append(curr_pt)

    if len(current_path) >= 2:
        split_paths.append(current_path)

    return split_paths


def extract_contour_geojson(raw_arr_k, contours_config=None):
    if not contours_config:
        return {"type": "FeatureCollection", "features": []}

    try:
        frame_h, frame_w = raw_arr_k.shape
        
        smoothed = cv2.GaussianBlur(raw_arr_k.astype(np.float32), (5, 5), 1.2)
        smoothed_flipped = np.flipud(smoothed)
        
        smoothed_cyclic = np.hstack([smoothed_flipped, smoothed_flipped[:, :1]])
        
        lon_step = 360.0 / frame_w
        lons = np.linspace(-180.0, 180.0 + lon_step, frame_w + 1)
        lats = np.linspace(-90.0, 90.0, frame_h)
            
        cont_gen = contourpy.contour_generator(x=lons, y=lats, z=smoothed_cyclic)
        features = []

        for c_def in contours_config:
            target_val = c_def["target"]
            lines = cont_gen.lines(target_val)
            
            segments = []
            for line_array in lines:
                if len(line_array) >= 2:
                    pts = []
                    for pt in line_array:
                        lng = float(pt[0])
                        lat = float(pt[1])
                        if lng > 180.0:
                            lng = 180.0
                        pts.append([round(lng, 4), round(lat, 4)])
                    
                    all_on_left = all(abs(p[0] - (-180.0)) < 0.01 for p in pts)
                    all_on_right = all(abs(p[0] - 180.0) < 0.01 for p in pts)
                    
                    if not all_on_left and not all_on_right:
                        segments.append(pts)

            if segments:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": segments},
                    "properties": {
                        "name": c_def["name"],
                        "color": c_def["color"],
                        "width": c_def["width"],
                        "opacity": c_def["opacity"]
                    }
                })

        if not features:
            print(f"  ⚠️ Note: 0 contour feature sets generated.")
            return {"type": "FeatureCollection", "features": []}

        print(f"  ✨ Generated {len(features)} contour feature set(s)")
        return {
            "type": "FeatureCollection",
            "features": features
        }
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return {"type": "FeatureCollection", "features": []}


def process_grib_to_array(grib_path, param_config):
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={'errors': 'ignore'})
    
    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})

    ds = ds.sortby('latitude', ascending=False)
    
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        ).sortby('longitude')

    grib_var = param_config["grib_param"]
    target_var = grib_var if grib_var in ds else list(ds.data_vars)[0]
    data_array = ds[target_var]

    raw_arr_k = np.squeeze(data_array.values)
    ds.close()

    contour_geojson = extract_contour_geojson(raw_arr_k, param_config.get("contours", []))

    min_val = param_config["min_val"]
    max_val = param_config["max_val"]

    arr = np.nan_to_num(raw_arr_k, copy=False, nan=min_val)
    np.clip(arr, min_val, max_val, out=arr)
    arr -= min_val
    arr /= (max_val - min_val)
    
    # 🌟 12-bit precision (4,096 levels -> finer than 0.01" accuracy) with LSB zeroing
    arr *= 4095.0
    arr_int = arr.astype(np.uint16)
    arr_int = (arr_int >> 4) << 4

    return arr_int, contour_geojson


def fetch_and_process_step(client, target_date, chosen_run, step, param_config, model_name):
    patterns = param_config["filename_patterns"]
    grib_file = patterns["grib"].format(model=model_name, param=param_config["id"], step=step)

    try:
        client.retrieve(
            date=target_date, 
            time=int(chosen_run), 
            step=step,
            type=param_config["type"], 
            levtype=param_config["levtype"], 
            param=[param_config["grib_param"]], 
            target=grib_file
        )
        if os.path.exists(grib_file):
            frame_arr, contour_geojson = process_grib_to_array(grib_file, param_config)
            try: os.remove(grib_file)
            except Exception: pass
            print(f"  ⚡ [{param_config['id']}] Processed F{step:03d}")
            return step, frame_arr, contour_geojson
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None, None


def build_spritesheet_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run):
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    frames_per_sheet = 10  # 10 horizontal frames per chunk (1 row of 10)

    chunks = []
    patterns = param_config["filename_patterns"]
    
    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_sheet)):
        chunk_frames = frame_arrays[i:i + frames_per_sheet]
        chunk_steps = steps_written[i:i + frames_per_sheet]
        
        num_cols = len(chunk_frames)
        sheet_w = frame_w * num_cols
        sheet_rows = 1
        sheet_h = frame_h
        
        spritesheet_arr = np.zeros((sheet_h, sheet_w), dtype=np.uint16)  # 🌟 uint16 buffer

        for idx, arr in enumerate(chunk_frames):
            x_start = idx * frame_w
            x_end = x_start + frame_w
            spritesheet_arr[0:frame_h, x_start:x_end] = arr

        spritesheet_filename = patterns["spritesheet"].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            chunk_idx=chunk_idx
        )
        
        chunks.append({
            "array": spritesheet_arr,
            "manifest_data": {
                "file": spritesheet_filename,
                "forecast_steps": chunk_steps,
                "columns": num_cols,
                "rows": 1,
                "sheet_width": sheet_w,
                "sheet_height": sheet_h
            }
        })

    return chunks, frame_w, frame_h


def upload_single_file(s3_client, bucket_name, filepath, filename):
    content_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    try:
        with open(filepath, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=filename,
                Body=f,
                ContentType=content_type
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

    print(f"\n☁️ Uploading {folder_path} assets to Backblaze B2 concurrently...")

    s3_client = boto3.client(
        service_name='s3',
        endpoint_url=f"https://{endpoint}",
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key
    )

    all_files = [
        fname for fname in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, fname))
    ]

    asset_files = [f for f in all_files if not f.endswith('manifest.json')]
    manifest_files = [f for f in all_files if f.endswith('manifest.json')]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, os.path.join(folder_path, fname), fname)
            for fname in asset_files
        ]
        concurrent.futures.wait(futures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, os.path.join(folder_path, fname), fname)
            for fname in manifest_files
        ]
        concurrent.futures.wait(futures)


def run_master_pipeline(selected_param_key="2t"):
    MODEL_NAME = "ecmwf"
    param_config = load_parameter_config(selected_param_key)
    patterns = param_config["filename_patterns"]

    output_dist_dir = f"run_{param_config['id']}"

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

    print(f"🌍 [{param_config['id']}] Model: {MODEL_NAME} | Param: {param_config['name']} | Run: {CHOSEN_RUN}z on {target_date}")

    for f in glob.glob(f"{MODEL_NAME}_{param_config['id']}_*.grib2"):
        try: os.remove(f)
        except Exception: pass

    client = Client(source="azure", model="ifs", resol="0p25")
    os.makedirs(output_dist_dir, exist_ok=True)

    results = {}
    contours_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, client, target_date, CHOSEN_RUN, step, param_config, MODEL_NAME): step
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
        print(f"❌ [{param_config['id']}] No frames processed. Exiting pipeline.")
        return

    populated_steps = {
        str(step): contours_dict[step] 
        for step in sorted_steps 
        if step in contours_dict and contours_dict[step] and len(contours_dict[step].get("features", [])) > 0
    }

    master_contours = {
        "model": MODEL_NAME,
        "parameter": param_config["id"],
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "steps": populated_steps
    }

    run_contour_filename = patterns["run_contours"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
    )
    with open(os.path.join(output_dist_dir, run_contour_filename), 'w') as f:
        json.dump(master_contours, f)

    latest_contour_filename = patterns["latest_contours"].format(
        model=MODEL_NAME, param=param_config["id"]
    )
    with open(os.path.join(output_dist_dir, latest_contour_filename), 'w') as f:
        json.dump(master_contours, f)

    chunks, frame_w, frame_h = build_spritesheet_chunks(
        frame_arrays, 
        steps_written, 
        model_name=MODEL_NAME, 
        param_config=param_config,
        target_date=target_date, 
        chosen_run=CHOSEN_RUN
    )

    manifest_chunks = []
    
    for chunk in chunks:
        filename = chunk["manifest_data"]["file"]
        filepath = os.path.join(output_dist_dir, filename)
        
        # 🌟 Gzip compress the 16-bit binary chunks
        if filename.endswith(".bin"):
            with open(filepath, "wb") as f:
                f.write(gzip.compress(chunk["array"].tobytes(), compresslevel=6))
        else:
            cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            
        manifest_chunks.append(chunk["manifest_data"])

    manifest = {
        "model": MODEL_NAME,
        "parameter": param_config["id"],
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "spritesheet_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": param_config["min_val"],
        "temp_max_k": param_config["max_val"],
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    run_manifest_filename = patterns["run_manifest"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
    )
    
    for m_fname in ["manifest.json", run_manifest_filename]:
        m_path = output_dist_dir / m_fname if isinstance(output_dist_dir, str) else os.path.join(output_dist_dir, m_fname)
        with open(m_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    print(f"\n🎉 [{param_config['id']}] Assets ready in {output_dist_dir}/")

    upload_to_b2_parallel(output_dist_dir)
    
    try:
        shutil.rmtree(output_dist_dir)
        print(f"  ✅ [{param_config['id']}] Cleanup complete.")
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Failed to delete temp directory: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_params = sys.argv[1:]
    else:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                target_params = list(data.get("parameters", {}).keys())
        else:
            target_params = ["2t"]

    print(f"🚀 Launching Pipeline for Parameters: {target_params}")
    
    max_param_workers = min(len(target_params), MAX_CONCURRENT_PARAMS)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_param_workers) as executor:
        futures = {executor.submit(run_master_pipeline, param): param for param in target_params}
        for future in concurrent.futures.as_completed(futures):
            param = futures[future]
            try:
                future.result()
                print(f"✅ Finished parameter: {param}")
            except Exception as e:
                print(f"❌ Error processing parameter '{param}': {e}")

    print("\n🎉 ALL PARAMETERS COMPLETED SUCCESSFULLY!")