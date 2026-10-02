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
import gzip  # 🌟 Built-in gzip for binary buffer support
import struct

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

MAX_FORECAST_HOURS = 360
FORECAST_STEPS = [h for h in range(0, MAX_FORECAST_HOURS + 1) if h % 3 == 0]

MAX_TEXTURE_SIZE = 4096

MAX_CONCURRENT_WORKERS = 4

MAX_CONCURRENT_PARAMS = 2

CONFIG_FILE_PATH = os.path.join("config", "parameters.json")
CONTOUR_COORD_SCALE = 1000


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


def extract_contour_geojson(raw_arr_k, contours_config=None, contour_settings=None):
    if not contours_config:
        return {"type": "FeatureCollection", "features": []}

    try:
        contour_settings = contour_settings or {}
        coarsen_stride = max(1, int(contour_settings.get("coarsen_stride", 1)))
        blur_kernel = tuple(contour_settings.get("blur_kernel", (5, 5)))
        blur_sigma = float(contour_settings.get("blur_sigma", 1.2))

        working_arr = raw_arr_k.astype(np.float32)
        if coarsen_stride > 1:
            working_arr = cv2.resize(
                working_arr,
                (max(1, working_arr.shape[1] // coarsen_stride), max(1, working_arr.shape[0] // coarsen_stride)),
                interpolation=cv2.INTER_AREA,
            )

        frame_h, frame_w = working_arr.shape
        smoothed = cv2.GaussianBlur(working_arr, blur_kernel, blur_sigma)
        smoothed_flipped = np.flipud(smoothed)
        smoothed_cyclic = np.hstack([smoothed_flipped, smoothed_flipped[:, :1]])

        lon_step = 360.0 / frame_w
        lons = np.linspace(-180.0, 180.0 + lon_step, frame_w + 1)
        lats = np.linspace(-90.0, 90.0, frame_h)
        cont_gen = contourpy.contour_generator(x=lons, y=lats, z=smoothed_cyclic)

        features = []
        interval = 6.0
        explicit_levels = []

        for c_def in contours_config:
            if c_def.get("dynamic"):
                interval = float(c_def.get("interval", 6.0))
                valid_mask = np.isfinite(working_arr) & (working_arr > 0)
                if not np.any(valid_mask):
                    explicit_levels = []
                    break

                valid_values = working_arr[valid_mask]
                data_min = float(np.nanmin(valid_values))
                data_max = float(np.nanmax(valid_values))
                if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                    start = math.ceil(data_min / interval) * interval
                    end = math.floor(data_max / interval) * interval
                    explicit_levels = [float(v) for v in np.arange(start, end + (interval * 0.5), interval)]
                break
            if "target" in c_def:
                explicit_levels.append(float(c_def["target"]))

        if not explicit_levels:
            return {"type": "FeatureCollection", "features": []}

        for target_val in sorted(set(explicit_levels)):
            lines = cont_gen.lines(target_val)
            segments = []
            for line_array in lines:
                if len(line_array) < 2:
                    continue
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
                name = str(target_val)
                for c_def in contours_config:
                    if c_def.get("dynamic") and c_def.get("unit") == "dam":
                        name = str(int(target_val)) if float(target_val).is_integer() else str(target_val)
                        break
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": segments},
                    "properties": {
                        "name": name,
                        "color": next((c_def["color"] for c_def in contours_config if c_def.get("dynamic") or c_def.get("target") == target_val), "#000000"),
                        "width": next((c_def["width"] for c_def in contours_config if c_def.get("dynamic") or c_def.get("target") == target_val), 1.6),
                        "opacity": next((c_def["opacity"] for c_def in contours_config if c_def.get("dynamic") or c_def.get("target") == target_val), 0.9)
                    }
                })

        if not features:
            print(f"  ⚠️ Note: 0 contour feature sets generated.")
            return {"type": "FeatureCollection", "features": []}

        print(f"  ✨ Generated {len(features)} contour feature set(s)")
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return {"type": "FeatureCollection", "features": []}


def encode_contours_binary(contours_by_step):
    """Encode quantized contour lines as per-line delta coordinates."""
    stream = bytearray(b"CTV1")
    stream.extend(struct.pack("<I", len(contours_by_step)))
    metadata_by_step = {}

    for step, collection in sorted(contours_by_step.items()):
        features = collection.get("features", [])
        stream.extend(struct.pack("<II", int(step), len(features)))
        metadata = []
        for feature in features:
            properties = feature.get("properties", {})
            metadata.append({
                "name": properties.get("name", ""),
                "color": properties.get("color", "#000000"),
                "width": properties.get("width", 1.6),
                "opacity": properties.get("opacity", 0.9),
            })
            lines = feature.get("geometry", {}).get("coordinates", [])
            stream.extend(struct.pack("<I", len(lines)))
            for line in lines:
                points = [
                    (round(point[0] * CONTOUR_COORD_SCALE), round(point[1] * CONTOUR_COORD_SCALE))
                    for point in line
                ]
                stream.extend(struct.pack("<I", len(points)))
                previous_x = previous_y = 0
                for point_x, point_y in points:
                    stream.extend(struct.pack("<ii", point_x - previous_x, point_y - previous_y))
                    previous_x, previous_y = point_x, point_y
        metadata_by_step[str(step)] = metadata

    return bytes(stream), metadata_by_step


def normalize_array(raw_arr, param_config):
    """
    🌟 DYNAMIC UNIVERSAL NORMALIZER
    Dynamically scales raw GRIB values into 8-bit image bytes based on parameters.json
    """
    scaling = param_config.get("scaling", {})
    mode = scaling.get("mode", "linear")

    # 🌟 Fully dynamic piecewise breakpoint interpolation via NumPy C-core
    if mode == "piecewise" and "val_points" in scaling and "byte_points" in scaling:
        multiplier = scaling.get("unit_multiplier", 1.0)
        v = np.nan_to_num(raw_arr, nan=0.0) * multiplier
        val_pts = scaling["val_points"]
        byte_pts = scaling["byte_points"]

        return np.interp(v, val_pts, byte_pts).astype(np.uint8)

    else:
        # Standard Linear Scaling (Temperature, Wind, Pressure, etc.)
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))
        
        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)


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

    if param_config.get("unit") == "dam" or str(param_config.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
        raw_arr_k = raw_arr_k / 98.0665

    contour_geojson = extract_contour_geojson(
        raw_arr_k,
        param_config.get("contours", []),
        param_config.get("contours_settings")
    )

    # 🌟 Dynamic normalization based on JSON config
    arr_8bit = normalize_array(raw_arr_k, param_config)

    return arr_8bit, contour_geojson


def fetch_and_process_step(client, target_date, chosen_run, step, param_config, model_name):
    patterns = param_config["filename_patterns"]
    grib_file = patterns["grib"].format(model=model_name, param=param_config["id"], step=step)

    retrieve_kwargs = {
        "date": target_date,
        "time": int(chosen_run),
        "step": step,
        "type": param_config.get("type", "fc"),
        "levtype": param_config.get("levtype", "sfc"),
        "param": [param_config["grib_param"]],
        "target": grib_file
    }

    if "levelist" in param_config:
        retrieve_kwargs["levelist"] = param_config["levelist"]

    try:
        client.retrieve(**retrieve_kwargs)
        if os.path.exists(grib_file):
            contour_source = param_config.get("contour_source")
            raster_config = {**param_config, "contours": []} if contour_source else param_config
            frame_arr, contour_geojson = process_grib_to_array(grib_file, raster_config)
            try: os.remove(grib_file)
            except Exception: pass

            if contour_source:
                contour_grib_file = patterns["grib"].format(
                    model=model_name,
                    param=f"{param_config['id']}_z500",
                    step=step
                )
                contour_retrieve_kwargs = dict(retrieve_kwargs)
                contour_retrieve_kwargs.update({
                    "param": [contour_source["grib_param"]],
                    "levtype": contour_source.get("levtype", param_config.get("levtype", "pl")),
                    "target": contour_grib_file
                })
                if "levelist" in contour_source:
                    contour_retrieve_kwargs["levelist"] = contour_source["levelist"]

                try:
                    client.retrieve(**contour_retrieve_kwargs)
                    if os.path.exists(contour_grib_file):
                        contour_param_config = {**param_config, **contour_source}
                        _, contour_geojson = process_grib_to_array(contour_grib_file, contour_param_config)
                except Exception as e:
                    print(f"  ❌ [{param_config['id']}] Height contour error F{step:03d}: {e}")
                finally:
                    if os.path.exists(contour_grib_file):
                        try: os.remove(contour_grib_file)
                        except Exception: pass

            print(f"  ⚡ [{param_config['id']}] Processed F{step:03d}")
            return step, frame_arr, contour_geojson
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None, None


def build_volume_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run):
    """
    🌟 Builds contiguous 3D binary time-volume chunks [T, H, W]
    Preserves native resolution without 2D sprite edge bleeding.
    """
    if not frame_arrays:
        return [], 0, 0

    frame_h, frame_w = frame_arrays[0].shape
    frames_per_volume = 10  # 10 time frames per binary chunk

    chunks = []
    patterns = param_config["filename_patterns"]
    pattern_key = "volume" if "volume" in patterns else "spritesheet"
    
    for chunk_idx, i in enumerate(range(0, len(frame_arrays), frames_per_volume)):
        chunk_frames = frame_arrays[i:i + frames_per_volume]
        chunk_steps = steps_written[i:i + frames_per_volume]
        
        # Contiguous 3D stack of frames [T, H, W]
        volume_arr = np.stack(chunk_frames, axis=0).astype(np.uint8)

        volume_filename = patterns[pattern_key].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            chunk_idx=chunk_idx
        )
        
        chunks.append({
            "array": volume_arr,
            "manifest_data": {
                "file": volume_filename,
                "forecast_steps": chunk_steps,
                "frame_count": len(chunk_frames)
            }
        })

    return chunks, frame_w, frame_h


def upload_single_file(s3_client, bucket_name, filepath, filename):
    content_type = "application/json" if filename.endswith(".json") else ("application/octet-stream" if filename.endswith(".bin") else "image/png")
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
    contour_binary, contour_metadata = encode_contours_binary(populated_steps)
    manifest_steps = {
        step: {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": []},
                    "properties": properties,
                }
                for properties in contour_metadata[step]
            ],
        }
        for step in contour_metadata
    }
    master_contours["steps"] = manifest_steps

    run_contour_filename = patterns["run_contours"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
    )
    run_binary_filename = run_contour_filename.replace(".json", ".bin.gz")
    master_contours["binary"] = {
        "format": "CTV1",
        "scale": CONTOUR_COORD_SCALE,
        "file": run_binary_filename,
    }
    with open(os.path.join(output_dist_dir, run_contour_filename), 'w') as f:
        json.dump(master_contours, f)
    with open(os.path.join(output_dist_dir, run_binary_filename), "wb") as f:
        f.write(gzip.compress(contour_binary, compresslevel=6))

    latest_contour_filename = patterns["latest_contours"].format(
        model=MODEL_NAME, param=param_config["id"]
    )
    latest_binary_filename = latest_contour_filename.replace(".json", ".bin.gz")
    master_contours["binary"]["file"] = latest_binary_filename
    with open(os.path.join(output_dist_dir, latest_contour_filename), 'w') as f:
        json.dump(master_contours, f)
    with open(os.path.join(output_dist_dir, latest_binary_filename), "wb") as f:
        f.write(gzip.compress(contour_binary, compresslevel=6))

    chunks, frame_w, frame_h = build_volume_chunks(
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
        
        # 🌟 Writes Gzip-compressed raw binary buffer
        if filename.endswith(".bin"):
            with open(filepath, "wb") as f:
                f.write(gzip.compress(chunk["array"].tobytes(), compresslevel=9))
        elif filename.endswith(".webp"):
            cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_WEBP_QUALITY), 101])
        else:
            cv2.imwrite(filepath, chunk["array"], [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            
        manifest_chunks.append(chunk["manifest_data"])

    manifest = {
        "model": MODEL_NAME,
        "parameter": param_config["id"],
        "name": param_config.get("name", param_config["id"]),
        "unit": param_config.get("unit", ""),
        "scaling": param_config.get("scaling", {}),
        "run": f"{CHOSEN_RUN}z",
        "date": target_date,
        "init_time": init_time_iso,
        "type": "volume_chunked",
        "total_frames": len(steps_written),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "temp_min_k": param_config.get("min_val", param_config.get("scaling", {}).get("min_val", 0.0)),
        "temp_max_k": param_config.get("max_val", param_config.get("scaling", {}).get("max_val", 255.0)),
        "chunks": manifest_chunks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    }

    run_manifest_filename = patterns["run_manifest"].format(
        model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
    )
    
    manifest_files_to_write = ["manifest.json", run_manifest_filename]

    if "latest_manifest" in patterns:
        latest_manifest_filename = patterns["latest_manifest"].format(
            model=MODEL_NAME, param=param_config["id"]
        )
        manifest_files_to_write.append(latest_manifest_filename)

    for m_fname in manifest_files_to_write:
        m_path = os.path.join(output_dist_dir, m_fname)
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
    
    batch_size = MAX_CONCURRENT_PARAMS

    for i in range(0, len(target_params), batch_size):
        batch = target_params[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = math.ceil(len(target_params) / batch_size)
        
        print(f"\n📦 [Batch {batch_num}/{total_batches}] Running {len(batch)} parameter(s) concurrently: {batch}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(run_master_pipeline, param): param for param in batch}
            for future in concurrent.futures.as_completed(futures):
                param = futures[future]
                try:
                    future.result()
                    print(f"✅ Finished parameter: {param}")
                except Exception as e:
                    print(f"❌ Error processing parameter '{param}': {e}")

    print("\n🎉 ALL PARAMETERS COMPLETED SUCCESSFULLY!")
