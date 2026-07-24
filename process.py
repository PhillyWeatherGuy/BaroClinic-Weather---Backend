from plot_maps import plot_weather_map, export_weather_data 
import os 
import datetime 
import glob 
import time 
import math  # Added for grid calculations
from ecmwf.opendata import Client 
from PIL import Image 

# Focused exclusively on 2m Temperature for testing 
WEATHER_PARAMETERS = { 
    "2t": {"param_key": ["2t"], "levtype": "sfc", "levelist": None, "local_prefix": "t2m", "start_hour": 0} 
} 

OUTPUT_MAPS_DIR = "output_maps" 
OUTPUT_DIST_DIR = "run_conus" # Set to run_conus to pass through git filters 
WEBP_QUALITY = 75  # Section 10 standard WebP compression choice 

def run_master_data_downloader(): 
    """ 
    Hour-by-Hour Data Engine. Downloads 2m Temp for a single  
    forecast hour before moving to the next. Concludes with a stitching phase. 
    """ 
    now_utc = datetime.datetime.now(datetime.timezone.utc) 
    current_hour = now_utc.hour 
     
    if current_hour >= 18: 
        CHOSEN_RUN = "18" 
        target_date = now_utc.strftime("%Y%m%d") 
        fallback_date = target_date 
    elif current_hour >= 12: 
        CHOSEN_RUN = "12" 
        target_date = now_utc.strftime("%Y%m%d") 
        fallback_date = target_date 
    elif current_hour >= 6: 
        CHOSEN_RUN = "06" 
        target_date = now_utc.strftime("%Y%m%d") 
        fallback_date = target_date 
    else: 
        CHOSEN_RUN = "18" 
        target_date = (now_utc - datetime.timedelta(days=1)).strftime("%Y%m%d") 
        fallback_date = target_date 

    MAX_FORECAST_HOURS = 45 
    is_fallback_active = False 

    print(f"🌍 Current UTC Hour: {current_hour:02d}z | Automatically selected run: {CHOSEN_RUN}z on {target_date}") 
    print("🧹 Purging old local GRIB files...") 
    for old_file in glob.glob("ecmwf_t2m_*.grib2"): 
        try: os.remove(old_file) 
        except Exception: pass 

    # Initialize Client with retries 
    client = None 
    max_init_retries = 5 
    for attempt in range(1, max_init_retries + 1): 
        try: 
            print(f"🔌 Connecting to ECMWF Azure Mirror (Attempt {attempt}/{max_init_retries})...") 
            client = Client(source="azure", model="ifs", resol="0p25") 
            print("🚀 Successfully connected and obtained Azure credentials!") 
            break 
        except Exception as conn_err: 
            print(f"⚠️ Connection failed: {conn_err}") 
            if attempt < max_init_retries: 
                sleep_time = attempt * 5 
                print(f"😴 Waiting {sleep_time} seconds before retrying...") 
                time.sleep(sleep_time) 
            else: 
                print("❌ Could not connect to ECMWF client. Exiting.") 
                raise conn_err 
     
    forecast_steps = [] 
    for h in range(0, MAX_FORECAST_HOURS + 1): 
        if h <= 144 and h % 3 == 0: 
            forecast_steps.append(h) 
        elif h > 144 and h % 6 == 0: 
            forecast_steps.append(h) 

    processed_steps = [] 

    # Loop through hours sequentially 
    for step in forecast_steps: 
        print(f"\n⏰ --------------------------------------------------") 
        print(f"🚀 PROCESSING FORECAST HOUR: F{step:03d}") 
        print(f"--------------------------------------------------") 
         
        hour_download_count = 0 
        current_search_date = fallback_date if is_fallback_active else target_date 
        config = WEATHER_PARAMETERS["2t"] 
        target_filename = f"ecmwf_t2m_{step:03d}.grib2" 
         
        try: 
            retrieve_kwargs = { 
                "date": current_search_date, 
                "time": CHOSEN_RUN, 
                "step": step, 
                "type": "fc", 
                "levtype": config["levtype"], 
                "param": config["param_key"], 
                "target": target_filename 
            } 
             
            client.retrieve(**retrieve_kwargs) 
            print(f"    🟢 [2T] saved -> {target_filename}") 
            hour_download_count += 1 
             
        except Exception as e: 
            if step in [0, 3] and not is_fallback_active: 
                adjusted_dt = datetime.datetime.strptime(target_date, "%Y%m%d") - datetime.timedelta(days=1) 
                fallback_date = adjusted_dt.strftime("%Y%m%d") 
                print(f"    ⚠️ Run '{target_date} {CHOSEN_RUN}z' incomplete. Pivoting to fallback date: '{fallback_date}'...") 
                is_fallback_active = True 
                current_search_date = fallback_date 
                try: 
                    retrieve_kwargs["date"] = fallback_date 
                    client.retrieve(**retrieve_kwargs) 
                    print(f"    🟢 Fallback success! Saved -> {target_filename}") 
                    hour_download_count += 1 
                except Exception as fallback_err: 
                    print(f"    × ⚠️ Fallback download failed: {fallback_err}") 

            if hour_download_count == 0: 
                print(f"    🛑 [2T] frame not available on server.") 
                if os.path.exists(target_filename): 
                    try: os.remove(target_filename) 
                    except Exception: pass 

        if hour_download_count == 0: 
            print(f"\n✨ Hit the ungenerated data boundary at F{step:03d}. Wrapping up feed.") 
            break 
             
        processed_steps.append(step) 
             
        # 🎨 PLOT TRIGGER 
        if os.path.exists(target_filename): 
            print(f"    🎨 Generating 2m Temperature map for step F{step:03d}...") 
            try: 
                plot_weather_map(target_filename, step, sector_name='CONUS', theme='DARK') 
                export_weather_data(target_filename, step) 
            except Exception as map_err: 
                print(f"    × ⚠️ Mapping failed for step F{step:03d}: {map_err}") 

    # 🧵 SPRITESHEET STITCHING PHASE (GRID METHOD) 
    print("\n🧵 --------------------------------------------------") 
    print("⚡ STARTING AUTOMATED GRID SPRITESHEET STITCHING") 
    print("--------------------------------------------------") 
     
    os.makedirs(OUTPUT_DIST_DIR, exist_ok=True) 
    frames = [] 
     
    for step in processed_steps: 
        img_filename = f"temp_CONUS_{step:03d}.webp" 
        img_path = os.path.join(OUTPUT_MAPS_DIR, img_filename) 
         
        if os.path.exists(img_path): 
            try: 
                img = Image.open(img_path) 
                frames.append(img) 
            except Exception as open_err: 
                print(f"    ⚠️ Failed to open frame {img_filename}: {open_err}") 
                 
    if frames: 
        print(f"📦 Assembling {len(frames)} frames into a 2D Grid for 2m Temperature...") 
        frame_width, frame_height = frames[0].size 
        
        # 🧮 GRID CALCULATOR
        MAX_WEBP_DIM = 16383
        
        # Find maximum frames we can fit side-by-side before hitting the limit
        max_cols = MAX_WEBP_DIM // frame_width
        if max_cols == 0: max_cols = 1 # Edge case safeguard
        
        # Calculate actual columns and rows needed
        cols = min(len(frames), max_cols)
        rows = math.ceil(len(frames) / cols)
        
        total_width = cols * frame_width
        total_height = rows * frame_height
        
        # 🛡️ GRID SAFEGUARD
        # Triggers if the Y-axis exceeds the WebP limit (useful for massive runs like 240 hrs)
        max_dimension = max(total_width, total_height)
        if max_dimension > MAX_WEBP_DIM:
            scale_factor = MAX_WEBP_DIM / max_dimension
            new_width = int(frame_width * scale_factor)
            new_height = int(frame_height * scale_factor)
            print(f"⚠️ Grid dimension ({max_dimension}px) exceeds WebP limit. Auto-scaling frames to {new_width}x{new_height}...")
            
            frames = [f.resize((new_width, new_height), Image.Resampling.LANCZOS) for f in frames]
            frame_width, frame_height = new_width, new_height
            
            # Recalculate dimensions for the new scaled frames
            max_cols = MAX_WEBP_DIM // frame_width
            cols = min(len(frames), max_cols)
            rows = math.ceil(len(frames) / cols)
            total_width = cols * frame_width
            total_height = rows * frame_height
         
        spritesheet = Image.new("RGBA", (total_width, total_height)) 
         
        # Paste frames into their correct grid coordinates
        for idx, frame in enumerate(frames): 
            col = idx % cols
            row = idx // cols
            x_offset = col * frame_width 
            y_offset = row * frame_height
            spritesheet.paste(frame, (x_offset, y_offset)) 
             
        # 🧹 CLEANUP PASS: Sweep all legacy outputs
        print("🧹 Sweeping old distribution hub files to ensure clean git tracking...") 
        old_files_to_remove = (
            glob.glob(os.path.join(OUTPUT_DIST_DIR, "t2m_dark_sheet_*_*.webp")) + # New Grid format
            glob.glob(os.path.join(OUTPUT_DIST_DIR, "t2m_dark_sheet_*.webp")) +   # Legacy format
            glob.glob(os.path.join(OUTPUT_DIST_DIR, "t2m_dark_sheet.json"))       # Legacy JSON
        )
        for old_file in old_files_to_remove: 
            try: 
                os.remove(old_file) 
                print(f"    🗑️ Removed legacy file: {os.path.basename(old_file)}") 
            except Exception: pass 
             
        # Target format: t2m_dark_sheet_[totalFrames]_[columns].webp
        sheet_path = os.path.join(OUTPUT_DIST_DIR, f"t2m_dark_sheet_{len(frames)}_{cols}.webp") 
        spritesheet.save(sheet_path, "WEBP", quality=WEBP_QUALITY) 
        print(f"    ✅ Fresh 2D grid spritesheet written successfully -> {sheet_path}") 
         
        # 🧹 Disk Clean up to keep repository completely safe from bloat 
        print("    🧹 Cleaning up temporary loop frames and GRIB files...") 
        for step in processed_steps: 
            img_filename = f"temp_CONUS_{step:03d}.webp" 
            try: os.remove(os.path.join(OUTPUT_MAPS_DIR, img_filename)) 
            except Exception: pass 
            try: os.remove(f"ecmwf_t2m_{step:03d}.grib2") 
            except Exception: pass 
    else: 
        print("📁 No frames were found to stitch.") 

    print("\n🎉 Live-stream data downloader and 2D Grid build complete!") 

if __name__ == "__main__": 
    run_master_data_downloader()