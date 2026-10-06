import os
import shutil
import datetime
import glob
import json
import math
import sys
import re
import concurrent.futures
import numpy as np
import xarray as xr
import cv2
from ecmwf.opendata import Client
import rioxarray
from rasterio.enums import Resampling
import boto3
import contourpy
import gzip
import struct

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"

# ECMWF open data (0p25 IFS): 3-hourly to 144h, then 6-hourly to 360h.
# Requesting anything else (e.g. 147, 153, ...) just returns a 404.
FORECAST_STEPS = list(range(0, 145, 3)) + list(range(150, 361, 6))

MAX_TEXTURE_SIZE = 4096

MAX_CONCURRENT_WORKERS = 4

MAX_CONCURRENT_PARAMS = 2

CONFIG_FILE_PATH = os.path.join("config", "parameters.json")

CONTOUR_COORD_SCALE = 1000          # stored as integer thousandths of a degree
CONTOUR_FORMAT = "CTV2"
INT16_MAX = 32767

CLIMATOLOGY_CACHE = {}


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


# ---------------------------------------------------------------------------
# Contours: vectorized extraction + CTV2 binary packing
#
# Per level the payload is (all little-endian):
#   uint32  n_lines
#   uint32  lens[n_lines]        points per line
#   int32   start_x[n_lines]     absolute first vertex of each line (x1000)
#   int32   start_y[n_lines]
#   int16   dx[sum(lens)-n]      per-vertex deltas, all X first ...
#   int16   dy[sum(lens)-n]      ... then all Y
# Every block is a multiple of 4 bytes, so typed arrays stay aligned in JS.
# ---------------------------------------------------------------------------

def _pack_level(cont_gen, level):
    """Contour one level and return packed bytes, or None if nothing to draw."""
    pts_list, off_list = cont_gen.lines(level)
    pts, off = pts_list[0], off_list[0]
    if pts is None or off is None or len(off) < 2:
        return None

    off = off.astype(np.intp)
    lens = np.diff(off)
    starts = off[:-1]

    # Clamp the wrap-around column to 180, then drop lines lying entirely on a seam.
    x = np.minimum(pts[:, 0], 180.0)
    seam = (np.maximum.reduceat(x, starts) < -179.99) | (np.minimum.reduceat(x, starts) > 179.99)

    q = np.empty((len(pts), 2), dtype=np.int32)
    q[:, 0] = np.rint(x * CONTOUR_COORD_SCALE)
    q[:, 1] = np.rint(pts[:, 1] * CONTOUR_COORD_SCALE)

    keep_line = ~seam
    if not keep_line.all():
        q = q[np.repeat(keep_line, lens)]
        lens = lens[keep_line]
    if len(lens) == 0:
        return None

    n = len(q)
    is_start = np.zeros(n, dtype=bool)
    is_start[np.cumsum(lens) - lens] = True

    # Safety net: if two neighbouring vertices are too far apart for int16,
    # start a new line there instead of overflowing.
    if n > 1:
        is_start[1:] |= (np.abs(np.diff(q, axis=0)) > INT16_MAX).any(axis=1)

    seg_starts = np.flatnonzero(is_start)
    seg_lens = np.diff(np.append(seg_starts, n))

    keep_seg = seg_lens >= 2
    if not keep_seg.all():
        q = q[np.repeat(keep_seg, seg_lens)]
        seg_lens = seg_lens[keep_seg]
        seg_starts = np.cumsum(seg_lens) - seg_lens
        if len(seg_lens) == 0:
            return None
        is_start = np.zeros(len(q), dtype=bool)
        is_start[seg_starts] = True

    deltas = np.diff(q, axis=0)
    inner = ~is_start[1:]
    dx = deltas[inner, 0].astype("<i2")
    dy = deltas[inner, 1].astype("<i2")

    return b"".join((
        struct.pack("<I", len(seg_lens)),
        seg_lens.astype("<u4").tobytes(),
        q[seg_starts, 0].astype("<i4").tobytes(),
        q[seg_starts, 1].astype("<i4").tobytes(),
        dx.tobytes(),
        dy.tobytes(),
    ))


def _level_style(level, contours_config):
    """Name/color/width/opacity for a level (same rules as before, computed once per level)."""
    name = str(level)
    unit = None
    for c_def in contours_config:
        # Formats integers cleanly for dam, mb, and hPa
        if c_def.get("dynamic") and c_def.get("unit") in {"dam", "mb", "hPa"}:
            unit = c_def.get("unit")
            name = str(int(level)) if float(level).is_integer() else str(level)
            break

    match = next(
        (c for c in contours_config if c.get("dynamic") or c.get("target") == level),
        {},
    )
    return {
        "name": name,
        "unit": unit or match.get("unit", ""),
        "color": match.get("color", "#000000"),
        "width": match.get("width", 1.6),
        "opacity": match.get("opacity", 0.9),
    }


def extract_contour_levels(raw_arr_k, contours_config=None, contour_settings=None):
    """
    Returns a list of {"props": {...}, "payload": bytes}, one per contour level that has lines.
    No GeoJSON and no per-point Python loops.
    """
    if not contours_config:
        return []

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
        cont_gen = contourpy.contour_generator(
            x=lons, y=lats, z=smoothed_cyclic,
            line_type=contourpy.LineType.ChunkCombinedOffset,
        )

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
            return []

        levels = []
        for target_val in sorted(set(explicit_levels)):
            payload = _pack_level(cont_gen, target_val)
            if payload is not None:
                levels.append({"props": _level_style(target_val, contours_config), "payload": payload})

        if not levels:
            print("  ⚠️ Note: 0 contour feature sets generated.")
        else:
            print(f"  ✨ Generated {len(levels)} contour feature set(s)")
        return levels
    except Exception as e:
        print(f"  ❌ Contour extraction exception: {e}")
        return []


def pack_step(levels):
    """Concatenate one step's level payloads (runs inside the worker thread)."""
    return {
        "blob": b"".join(l["payload"] for l in levels),
        "props": [l["props"] for l in levels],
    }


def assemble_contour_file(steps):
    """steps: {step_int: {"blob": bytes, "props": [...]}} -> full CTV2 file bytes."""
    parts = [CONTOUR_FORMAT.encode("ascii"), struct.pack("<I", len(steps))]
    for step in sorted(steps):
        entry = steps[step]
        parts.append(struct.pack("<III", int(step), len(entry["props"]), len(entry["blob"])))
        parts.append(entry["blob"])
    return b"".join(parts)


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
        # Standard Linear Scaling (Temperature, Wind, Pressure, Anomaly, etc.)
        min_v = scaling.get("min_val", param_config.get("min_val", 0.0))
        max_v = scaling.get("max_val", param_config.get("max_val", 255.0))

        arr = np.nan_to_num(raw_arr, copy=False, nan=min_v)
        np.clip(arr, min_v, max_v, out=arr)
        arr -= min_v
        arr /= (max_v - min_v)
        arr *= 255.0
        return arr.astype(np.uint8)


def get_climatology_grid(param_id, target_date=None, target_shape=(721, 1440), unit="m"):
    """
    🌟 Loads 30-year climatology baseline grid (Option 1).
    Looks in 'climatology/' or 'config/climatology/' for NetCDF (.nc) or NumPy (.npy/.npz).
    Caches in memory across threads. Falls back to zonal-mean if no file is found yet.
    """
    month = int(target_date[4:6]) if target_date and len(target_date) >= 6 else 10
    cache_key = f"{param_id}_{month}_{unit}"
    if cache_key in CLIMATOLOGY_CACHE:
        return CLIMATOLOGY_CACHE[cache_key]

    clim_dir = os.path.join("config", "climatology")
    if not os.path.exists(clim_dir):
        clim_dir = "climatology"

    candidate_files = [
        os.path.join(clim_dir, "z500_clim.nc"),
        os.path.join(clim_dir, "era5_z500_climatology.nc"),
        os.path.join(clim_dir, "z500_climatology.nc"),
        os.path.join(clim_dir, f"z500_clim_{month:02d}.npy"),
        os.path.join(clim_dir, "z500_clim.npy"),
        os.path.join(clim_dir, "z500_clim.npz"),
    ]

    for c_path in candidate_files:
        if os.path.exists(c_path):
            try:
                if c_path.endswith(".nc"):
                    with xr.open_dataset(c_path) as ds:
                        var = next((v for v in ["z", "gh", "z500", "hgt"] if v in ds), list(ds.data_vars)[0])
                        da = ds[var]
                        if "month" in da.coords:
                            da = da.sel(month=month)
                        grid = np.squeeze(da.values).astype(np.float32)
                elif c_path.endswith(".npz"):
                    with np.load(c_path) as data:
                        grid = data[f"month_{month:02d}"] if f"month_{month:02d}" in data else data[list(data.keys())[0]]
                elif c_path.endswith(".npy"):
                    grid = np.load(c_path).astype(np.float32)

                # Resize if climatology grid resolution differs from IFS 0.25 (721x1440)
                if grid.shape != target_shape:
                    grid = cv2.resize(grid, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

                # Convert geopotential to height if in m^2/s^2 (> 20000)
                if np.nanmean(grid) > 20000.0:
                    scale = 98.0665 if unit == "dam" else 9.80665
                    grid /= scale
                elif unit == "dam" and np.nanmean(grid) > 1000.0:
                    grid /= 10.0

                CLIMATOLOGY_CACHE[cache_key] = grid
                print(f"  📖 Loaded 30-year climatology baseline from: {c_path}")
                return grid
            except Exception as e:
                print(f"  ⚠️ Error loading climatology file {c_path}: {e}")

    return None


def process_thickness_grib(grib_path, contour_config):
    """
    🌟 Computes 1000-500mb thickness in decameters (dam): (Z500 - Z1000) / 98.0665
    """
    ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={'errors': 'ignore'})

    if 'lon' in ds.coords:
        ds = ds.rename({'lon': 'longitude'})
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude'})

    ds = ds.sortby('latitude', ascending=False)
    if ds.longitude.max() > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')

    target_var = "z" if "z" in ds else list(ds.data_vars)[0]
    da = ds[target_var]

    # Find the pressure level coordinate (typically 'isobaricInhPa')
    level_coord = next((c for c in da.coords if 'isobaric' in c.lower() or 'level' in c.lower()), None)

    if level_coord:
        z500 = da.sel({level_coord: 500}).values.squeeze()
        z1000 = da.sel({level_coord: 1000}).values.squeeze()
    else:
        raw_vals = da.values
        z500, z1000 = raw_vals[0].squeeze(), raw_vals[1].squeeze()

    ds.close()

    thickness_dam = (z500 - z1000) / 98.0665

    return extract_contour_levels(
        thickness_dam,
        contour_config.get("contours", []),
        contour_config.get("contours_settings")
    )


def process_grib_to_array(grib_path, param_config, want_raster=True, target_date=None):
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

    if param_config.get("category_mode"):
        return np.nan_to_num(raw_arr_k, nan=0.0).clip(0, 255).astype(np.uint8), []

    # 🌟 500mb Height Anomaly Calculation (Option 1: Z500 - Climatology)
    if "anom" in param_config.get("id", "").lower() or param_config.get("is_anomaly"):
        target_unit = param_config.get("unit", "m").lower()
        scale_div = 98.0665 if target_unit in {"dam"} else 9.80665
        hgt = raw_arr_k / scale_div

        clim = get_climatology_grid(param_config["id"], target_date, hgt.shape, unit=target_unit)
        if clim is None:
            # Graceful fallback: zonal-mean baseline until a climatology file is placed
            clim = np.tile(np.nanmean(hgt, axis=1, keepdims=True), (1, hgt.shape[1]))

        raw_arr_k = hgt - clim

    # 🌟 Unit Conversions for Contouring
    elif param_config.get("unit") == "dam" or str(param_config.get("grib_param", "")).lower() in {"z", "gh", "hgt"}:
        raw_arr_k = raw_arr_k / 98.0665
    elif param_config.get("unit") in {"mb", "hPa"} or str(param_config.get("grib_param", "")).lower() in {"msl", "mslp"}:
        raw_arr_k = raw_arr_k / 100.0

    contour_levels = extract_contour_levels(
        raw_arr_k,
        param_config.get("contours", []),
        param_config.get("contours_settings")
    )

    # Contour-only GRIBs (e.g. z500 or msl isobars) skip 8-bit image conversion
    arr_8bit = normalize_array(raw_arr_k, param_config) if want_raster else None

    return arr_8bit, contour_levels


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
            # Support multiple contour sources (contour_sources) or legacy single (contour_source)
            contour_sources = param_config.get("contour_sources")
            if not contour_sources and param_config.get("contour_source"):
                contour_sources = [param_config["contour_source"]]

            has_contours = bool(contour_sources)
            raster_config = {**param_config, "contours": []} if has_contours else param_config
            frame_arr, contour_levels = process_grib_to_array(grib_file, raster_config, target_date=target_date)
            contour_levels = list(contour_levels)
            ptype_arr = None

            try: os.remove(grib_file)
            except Exception: pass

            ptype_source = param_config.get("ptype_source")
            if ptype_source:
                ptype_file = f"{model_name}_{param_config['id']}_ptype_{step:03d}.grib2"
                ptype_retrieve_kwargs = dict(retrieve_kwargs)
                ptype_retrieve_kwargs.update({
                    "param": [ptype_source["grib_param"]],
                    "levtype": ptype_source.get("levtype", "sfc"),
                    "type": ptype_source.get("type", param_config.get("type", "fc")),
                    "target": ptype_file,
                })
                try:
                    client.retrieve(**ptype_retrieve_kwargs)
                    if os.path.exists(ptype_file):
                        ptype_config = {**ptype_source, "category_mode": True}
                        ptype_arr, _ = process_grib_to_array(ptype_file, ptype_config)
                except Exception as e:
                    print(f"  ⚠️ [{param_config['id']}] Precipitation type unavailable F{step:03d}: {e}")
                finally:
                    if os.path.exists(ptype_file):
                        try: os.remove(ptype_file)
                        except Exception: pass

            # Retrieve and extract all contour sources (e.g. MSLP + 1000-500mb Thickness)
            if contour_sources:
                for c_src in contour_sources:
                    c_id = c_src.get("id", c_src.get("grib_param", "contour"))
                    c_grib_file = patterns["grib"].format(
                        model=model_name,
                        param=f"{param_config['id']}_{c_id}",
                        step=step
                    )
                    c_kwargs = dict(retrieve_kwargs)
                    c_kwargs.update({
                        "param": [c_src["grib_param"]],
                        "levtype": c_src.get("levtype", "sfc"),
                        "target": c_grib_file
                    })
                    if "levelist" in c_src:
                        c_kwargs["levelist"] = c_src["levelist"]
                    elif "levelist" in c_kwargs:
                        del c_kwargs["levelist"]

                    try:
                        client.retrieve(**c_kwargs)
                        if os.path.exists(c_grib_file):
                            if c_src.get("type") == "thickness":
                                levels = process_thickness_grib(c_grib_file, c_src)
                            else:
                                c_cfg = {**param_config, **c_src}
                                _, levels = process_grib_to_array(
                                    c_grib_file, c_cfg, want_raster=False, target_date=target_date
                                )
                            contour_levels.extend(levels)
                    except Exception as e:
                        print(f"  ❌ [{param_config['id']}] Contour error ({c_id}) F{step:03d}: {e}")
                    finally:
                        if os.path.exists(c_grib_file):
                            try: os.remove(c_grib_file)
                            except Exception: pass

            # Pack contour bytes here, in the worker thread, so the end of the pipeline
            # only has to concatenate.
            packed = pack_step(contour_levels)

            print(f"  ⚡ [{param_config['id']}] Processed F{step:03d}")
            return step, frame_arr, packed, ptype_arr
    except Exception as e:
        print(f"  ❌ [{param_config['id']}] Error processing F{step:03d}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass
    return step, None, None, None


def build_volume_chunks(frame_arrays, steps_written, model_name, param_config, target_date, chosen_run, ptype_arrays=None):
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
        chunk_ptype = ptype_arrays[i:i + frames_per_volume] if ptype_arrays else None

        # Contiguous 3D stack of frames [T, H, W]
        volume_arr = np.stack(chunk_frames, axis=0).astype(np.uint8)

        volume_filename = patterns[pattern_key].format(
            model=model_name,
            param=param_config["id"],
            date=target_date,
            run=chosen_run,
            chunk_idx=chunk_idx
        )

        chunk_manifest = {
            "file": volume_filename,
            "forecast_steps": chunk_steps,
            "frame_count": len(chunk_frames)
        }
        chunk_entry = {
            "array": volume_arr,
            "manifest_data": chunk_manifest
        }
        if chunk_ptype and all(frame is not None for frame in chunk_ptype):
            ptype_filename = volume_filename.replace("_volume_", "_ptype_volume_")
            chunk_entry["ptype_array"] = np.stack(chunk_ptype, axis=0).astype(np.uint8)
            chunk_manifest["ptype_file"] = ptype_filename
        chunks.append(chunk_entry)

    return chunks, frame_w, frame_h


def upload_single_file(s3_client, bucket_name, filepath, filename):
    # 🌟 Guard against uploading empty or corrupt files (< 100 bytes)
    if os.path.getsize(filepath) < 100:
        print(f"  ⚠️ Skipping empty file (< 100 bytes): {filename}")
        return

    if filename.endswith(".json"):
        content_type = "application/json"
    elif filename.endswith(".gz"):
        content_type = "application/gzip"
    elif filename.endswith(".bin"):
        content_type = "application/octet-stream"
    else:
        content_type = "image/png"
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

    asset_files = [f for f in all_files if not f.endswith('.json')]
    json_files = [f for f in all_files if f.endswith('.json')]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, os.path.join(folder_path, fname), fname)
            for fname in asset_files
        ]
        concurrent.futures.wait(futures)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(upload_single_file, s3_client, bucket_name, os.path.join(folder_path, fname), fname)
            for fname in json_files
        ]
        concurrent.futures.wait(futures)


def upload_to_huggingface(folder_path, repo_id="PhillyWeatherGuy/baroclinic-model-data"):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set in environment. Skipping Hugging Face upload.")
        return

    print(f"\n🤗 Uploading {folder_path} assets to Hugging Face ({repo_id})...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload model grids: {folder_path}"
        )
        print(f"  ✅ Uploaded to Hugging Face: {folder_path}")
    except Exception as e:
        print(f"  ❌ Failed to upload to Hugging Face: {e}")


def prune_old_huggingface_runs(repo_id="PhillyWeatherGuy/baroclinic-model-data", days_to_keep=3):
    """
    🌟 Automatically prunes dated files older than `days_to_keep` days from Hugging Face.
    Leaves un-dated master pointers (manifest.json, latest contours) completely intact.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return

    cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    cutoff_int = int(cutoff_date.strftime("%Y%m%d"))

    print(f"\n🧹 Checking for Hugging Face runs older than {cutoff_date.strftime('%Y-%m-%d')} (>{days_to_keep} days)...")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        files_to_delete = []

        for fname in all_files:
            # Matches 8-digit date pattern: e.g. _20261005_
            match = re.search(r'_(\d{8})_', fname)
            if match:
                file_date_int = int(match.group(1))
                if file_date_int < cutoff_int:
                    files_to_delete.append(fname)

        if files_to_delete:
            print(f"  🗑️ Deleting {len(files_to_delete)} expired files older than {days_to_keep} days...")
            api.delete_files(
                repo_id=repo_id,
                repo_type="dataset",
                paths=files_to_delete,
                commit_message=f"Prune runs older than {days_to_keep} days"
            )
            print("  ✅ Pruning complete!")
        else:
            print("  ✨ No expired files found.")
    except Exception as e:
        print(f"  ⚠️ Could not prune old Hugging Face files: {e}")


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
    contour_steps = {}
    ptype_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        future_to_step = {
            executor.submit(fetch_and_process_step, client, target_date, CHOSEN_RUN, step, param_config, MODEL_NAME): step
            for step in FORECAST_STEPS
        }
        for future in concurrent.futures.as_completed(future_to_step):
            step, arr, packed, ptype_arr = future.result()
            if arr is not None:
                results[step] = arr
                if packed and packed["props"]:
                    contour_steps[step] = packed
                if ptype_arr is not None:
                    ptype_results[step] = ptype_arr

    sorted_steps = sorted(results.keys())
    frame_arrays = [results[s] for s in sorted_steps]
    ptype_arrays = [ptype_results.get(s) for s in sorted_steps]
    steps_written = sorted_steps

    if not frame_arrays:
        print(f"❌ [{param_config['id']}] No frames processed. Exiting pipeline.")
        return

    if contour_steps and "run_contours" in patterns:
        contour_binary = assemble_contour_file(contour_steps)
        compressed_contours = gzip.compress(contour_binary, compresslevel=6)

        if len(compressed_contours) >= 100:
            master_contours = {
                "model": MODEL_NAME,
                "parameter": param_config["id"],
                "run": f"{CHOSEN_RUN}z",
                "date": target_date,
                "steps": {
                    str(step): {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "geometry": {"type": "MultiLineString", "coordinates": []},
                                "properties": props,
                            }
                            for props in contour_steps[step]["props"]
                        ],
                    }
                    for step in sorted(contour_steps)
                },
            }

            run_contour_filename = patterns["run_contours"].format(
                model=MODEL_NAME, param=param_config["id"], date=target_date, run=CHOSEN_RUN.lower()
            )
            run_binary_filename = run_contour_filename.replace(".json", ".bin.gz")
            master_contours["binary"] = {
                "format": CONTOUR_FORMAT,
                "scale": CONTOUR_COORD_SCALE,
                "file": run_binary_filename,
            }

            print(f"  📦 [{param_config['id']}] Contours: {len(contour_binary)/1e6:.2f} MB raw -> {len(compressed_contours)/1e6:.2f} MB gzip")

            with open(os.path.join(output_dist_dir, run_contour_filename), 'w') as f:
                json.dump(master_contours, f, separators=(",", ":"))
            with open(os.path.join(output_dist_dir, run_binary_filename), "wb") as f:
                f.write(compressed_contours)

            if "latest_contours" in patterns:
                latest_contour_filename = patterns["latest_contours"].format(
                    model=MODEL_NAME, param=param_config["id"]
                )
                with open(os.path.join(output_dist_dir, latest_contour_filename), 'w') as f:
                    json.dump(master_contours, f, separators=(",", ":"))

    chunks, frame_w, frame_h = build_volume_chunks(
        frame_arrays,
        steps_written,
        model_name=MODEL_NAME,
        param_config=param_config,
        target_date=target_date,
        chosen_run=CHOSEN_RUN,
        ptype_arrays=ptype_arrays if ptype_results else None
    )

    manifest_chunks = []

    for chunk in chunks:
        filename = chunk["manifest_data"]["file"]
        filepath = os.path.join(output_dist_dir, filename)

        if filename.endswith(".bin"):
            with open(filepath, "wb") as f:
                f.write(gzip.compress(chunk["array"].tobytes(), compresslevel=9))
            ptype_filename = chunk["manifest_data"].get("ptype_file")
            if ptype_filename and "ptype_array" in chunk:
                with open(os.path.join(output_dist_dir, ptype_filename), "wb") as ptype_file:
                    ptype_file.write(gzip.compress(chunk["ptype_array"].tobytes(), compresslevel=9))
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

    upload_to_huggingface(output_dist_dir)
    # upload_to_b2_parallel(output_dist_dir)

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
    prune_old_huggingface_runs(days_to_keep=3)
