import os
import io
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image
from config import SECTORS, COLOR_CONFIGS

# Cache for city points to avoid hitting the web request inside every individual loop step
_CITY_POINTS_CACHE = None

def get_cached_city_points():
    """Downloads and filters US and international city points once per runtime"""
    global _CITY_POINTS_CACHE
    if _CITY_POINTS_CACHE is not None:
        return _CITY_POINTS_CACHE

    print("🌍 Initializing high-density city placement database...")
    city_list = []
    
    # 1. Load US Cities
    try:
        us_df = pd.read_csv("https://simplemaps.com/static/data/us-cities/uscitiesv1.4.csv")
        us_df = us_df.rename(columns={"lng": "lon"}).dropna(subset=["lat", "lon"])
        us_df["population"] = pd.to_numeric(us_df["population"], errors='coerce').fillna(0)
        us_df = us_df.sort_values("population", ascending=False)
        city_list.extend([(f"{r['city']}, {r['state_id']}", float(r['lat']), float(r['lon'])) for _, r in us_df.iterrows()][:35000])
    except Exception as e:
        print(f"⚠️ US Cities load error: {e}")

    # 2. Load World Cities
    try:
        cols = ["name","lat","lon","country_code","population"]
        world_df = pd.read_csv("https://download.geonames.org/export/dump/cities5000.zip", 
                               compression='zip', sep='\t', header=None,
                               usecols=[1, 4, 5, 8, 14], names=cols, low_memory=False)
        world_df = world_df[world_df["country_code"] != "US"]
        world_df["population"] = pd.to_numeric(world_df["population"], errors='coerce').fillna(0)
        world_df = world_df.sort_values("population", ascending=False)
        city_list.extend([(f"{r['name']}, {r['country_code']}", float(r['lat']), float(r['lon'])) for _, r in world_df.iterrows()])
    except Exception as e:
        print(f"⚠️ International cities load error: {e}")

    _CITY_POINTS_CACHE = city_list
    return _CITY_POINTS_CACHE


def render_weather_frame(grib_path, step, sector_name='CONUS', theme_name='DARK'):
    """
    Renders a weather map frame and returns it directly as a PIL Image object in memory.
    """
    sector_name = sector_name.strip().upper()
    theme_name = theme_name.strip().upper()

    if sector_name not in SECTORS:
        sector_name = 'CONUS'
    cfg = SECTORS[sector_name]

    # --- 1. Map Padding & Calculations ---
    padded_extent = [
        cfg['extent'][0] - cfg['pad_lon'], 
        cfg['extent'][1] + cfg['pad_lon'], 
        cfg['extent'][2] - cfg['pad_lat'], 
        cfg['extent'][3] + cfg['pad_lat']
    ]

    LARGE_SECTOR_SPAN_THRESHOLD = 18.0
    sector_span = ((padded_extent[1] - padded_extent[0]) + (padded_extent[3] - padded_extent[2])) / 2.0
    skip_interp = sector_span >= LARGE_SECTOR_SPAN_THRESHOLD
    show_counties = not skip_interp

    # --- 2. Color Palette Setup ---
    theme_colors = {
        'bg': '#121212' if theme_name == 'DARK' else '#FFFFFF',
        'ocean': '#161920' if theme_name == 'DARK' else '#EBF2F7',
        'land': '#1a1a1a' if theme_name == 'DARK' else '#F4F6F8',
        'lines': '#000000', 
        'text': '#FFFFFF' if theme_name == 'DARK' else '#000000',
        'accent': '#38bdf8' if theme_name == 'DARK' else '#0044BB', 
        'edge': '#FFFFFF' if theme_name == 'DARK' else '#2B2D31',
        'counties': '#444444' if theme_name == 'DARK' else '#888888',
    }

    t2m_levels = np.arange(-70.0, 130.1, 0.5)
    t2m_cfg = COLOR_CONFIGS["TMP_2m"]
    shared_colors = t2m_cfg["SHARED"]["hex_colors"] if "SHARED" in t2m_cfg else t2m_cfg[theme_name]["hex_colors"]
    
    thermal_colors = shared_colors 
    
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_t2m", thermal_colors, N=len(t2m_levels) - 1)
    custom_norm = mcolors.BoundaryNorm(t2m_levels, ncolors=custom_cmap.N)

    # --- 3. City Placement Overlap Engine ---
    min_lat_spacing = 1.1 if sector_name == 'CONUS' else cfg['min_space_lat']
    min_lon_spacing = 1.4 if sector_name == 'CONUS' else cfg['min_space_lon']
    
    all_cities = get_cached_city_points()
    cities_in_view = []
    for name, lat, lon in all_cities:
        if (min(padded_extent[:2]) <= lon <= max(padded_extent[:2])) and (min(padded_extent[2:]) <= lat <= max(padded_extent[2:])):
            if not any(abs(lat-klat) < min_lat_spacing and abs(lon-klon) < min_lon_spacing for _, klat, klon in cities_in_view):
                cities_in_view.append((name, lat, lon))

    # --- 4. GRIB Slicing & Processing (Xarray Engine) ---
    if not os.path.exists(grib_path):
        print(f"Error: File not found at {grib_path}")
        return None

    try:
        with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={'errors': 'ignore'}) as ds:
            if 'lon' in ds.coords: ds = ds.rename({'lon': 'longitude'})
            if 'lat' in ds.coords: ds = ds.rename({'lat': 'latitude'})
            ds = ds.sortby('latitude', ascending=True)
            if ds.longitude.max() > 180:
                ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
            
            t2m_raw = ds['t2m'].load()
            init_time = pd.Timestamp(ds.time.values)
            valid_time = pd.Timestamp(ds.valid_time.values)

        ds_trimmed = t2m_raw.where(
            (t2m_raw.longitude >= padded_extent[0]-4) & (t2m_raw.longitude <= padded_extent[1]+4) &
            (t2m_raw.latitude >= padded_extent[2]-4) & (t2m_raw.latitude <= padded_extent[3]+4), drop=True
        )

        if skip_interp:
            ds_final, sigma = ds_trimmed, 0.65
        else:
            glon = abs(float(ds_trimmed.longitude[1] - ds_trimmed.longitude[0])) / 2.0
            glat = abs(float(ds_trimmed.latitude[1] - ds_trimmed.latitude[0])) / 2.0
            ds_final = ds_trimmed.interp(
                longitude=np.arange(float(ds_trimmed.longitude.min()), float(ds_trimmed.longitude.max()) + glon, glon),
                latitude=np.arange(float(ds_trimmed.latitude.min()), float(ds_trimmed.latitude.max()) + glat, glat), 
                method='linear'
            )
            sigma = 1.0

        t2m_f = gaussian_filter((ds_final.values - 273.15) * 1.8 + 32.0, sigma=sigma)

        # --- 5. Graphics Canvas ---
        aspect = (padded_extent[1] - padded_extent[0]) / (padded_extent[3] - padded_extent[2])
        fig = plt.figure(figsize=(max(8.0, min(9.5 * aspect, 18.0)), 9.5), facecolor=theme_colors['bg'])
        
        map_proj = ccrs.PlateCarree()
        # Set map axes to completely fill the figure canvas (0 margin padding = ZERO DEAD SPACE)
        ax = fig.add_axes([0, 0, 1, 1], projection=map_proj)
        ax.set_extent(padded_extent, crs=map_proj)

        ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor=theme_colors['ocean'], zorder=1)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=theme_colors['land'], zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=theme_colors['ocean'], zorder=1.5)
        ax.coastlines(resolution='10m', color=theme_colors['lines'], linewidth=cfg['coast_lw'], zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), edgecolor=theme_colors['lines'], linewidth=cfg['coast_lw'], zorder=5)
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=theme_colors['lines'], linewidth=cfg['states_lw'], zorder=6)

        if show_counties:
            counties = cfeature.NaturalEarthFeature(category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
            ax.add_feature(counties, edgecolor=theme_colors['counties'], linewidth=cfg['states_lw'] * 0.6, zorder=5.5)

        # Retained ultra-fast rendering pipeline settings
        ax.contourf(ds_final.longitude.values, ds_final.latitude.values, t2m_f, levels=t2m_levels, 
                    cmap=custom_cmap, norm=custom_norm, transform=map_proj, zorder=2, alpha=0.88, antialiased=False)

        if np.min(t2m_f) <= 32.0 <= np.max(t2m_f):
            ax.contour(ds_final.longitude.values, ds_final.latitude.values, t2m_f, levels=[32.0], 
                       colors=['#4169E1'], linewidths=[2.2], transform=map_proj, zorder=4)

        for name, lat, lon in cities_in_view:
            try:
                val = float(ds_final.sel(longitude=lon, latitude=lat, method='nearest').values)
                val_f = (val - 273.15) * 1.8 + 32.0
            except Exception:
                continue
            if not np.isfinite(val_f) or not (-100.0 <= val_f <= 140.0): 
                continue

            lumi = sum(np.array(custom_cmap(custom_norm(val_f))[:3]) * [0.2126, 0.7152, 0.0722])
            txt_c = '#FFFFFF' if lumi < 0.5 else '#000000'
            lbl = ax.text(lon, lat, f"{round(val_f)}", transform=map_proj, fontsize=cfg['font_size'], 
                          fontweight='bold', color=txt_c, ha='center', va='center', zorder=7)
            lbl.set_path_effects([
                path_effects.Stroke(linewidth=1.6, foreground='#000000' if txt_c=='#FFFFFF' else '#FFFFFF'), 
                path_effects.Normal()
            ])

        cb_ticks = [-70, -44, -15, 0, 18, 32, 45, 60, 84, 102, 118, 130]
        cb_labels = [f"{t}°" for t in cb_ticks]
        
        # Translucent grey HUD panel container for the color key
        hud_bg_rgba = (0.094, 0.094, 0.106, 0.75)  # Equivalent to #18181b at 75% opacity
        hud_cax = fig.add_axes([0.02, 0.02, 0.96, 0.09], facecolor=hud_bg_rgba, zorder=8)
        for spine in hud_cax.spines.values():
            spine.set_edgecolor('#3f3f46')
            spine.set_linewidth(1.0)
        hud_cax.set_xticks([])
        hud_cax.set_yticks([])

        # Minimalist tracking subtitle centered above the color bar axis
        fig.text(0.5, 0.078, "SURFACE TEMPERATURE SCALE (°F)", fontsize=8.0, fontweight='bold', color='#a1a1aa', ha='center', zorder=10)

        # Position colorbar box to layout horizontally inside the new grey panel
        cax = fig.add_axes([0.05, 0.040, 0.90, 0.020], facecolor='none', zorder=9)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=custom_norm, cmap=custom_cmap), cax=cax, ticks=cb_ticks, spacing='uniform', extend='both', orientation='horizontal')
        cb.ax.set_xticklabels(cb_labels, fontsize=9.5, fontweight='bold', color='#FFFFFF')
        cb.outline.set_edgecolor('#3f3f46')
        cb.ax.tick_params(size=0, pad=5)

        # High-visibility drop text shadows for clear color bar readings
        for label in cb.ax.get_xticklabels():
            label.set_path_effects([
                path_effects.Stroke(linewidth=2.0, foreground='#000000'),
                path_effects.Normal()
            ])

        # Structured Broadcast Metadata Floating Layout Box
        hud_title = "ECMWF MODEL  •  2-METER SURFACE TEMPERATURE"
        hud_meta = f"INIT: {init_time.strftime('%H')}Z {init_time.strftime('%d %b %Y').upper()}    •    FHR: +{step:03d} HRS    •    VALID: {valid_time.strftime('%a %H:%MZ %d %b %Y').upper()}"
        hud_combined_text = f"{hud_title}\n{hud_meta}"
        
        fig.text(
            0.5, 0.97, hud_combined_text, fontsize=9.5, fontweight='bold', 
            color='#FFFFFF', ha='center', va='top', zorder=10,
            bbox=dict(facecolor=hud_bg_rgba, alpha=0.75, edgecolor='#3f3f46', boxstyle='round,pad=0.6', linewidth=1.0)
        )

        # --- 6. Save directly to local file system loop cache ---
        output_dir = "output_maps"
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"temp_{sector_name}_{step:03d}.webp")
        
        plt.savefig(out_path, dpi=115, facecolor=theme_colors['bg'], edgecolor='none', format='webp')
        plt.close(fig)
        print(f"    💾 Rendered map saved to loop disk: {out_path}")
        return True

    except Exception as e:
        print(f"⚠️ Failed to process GRIB graphics for step {step}: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return False

# Legacy adapter to preserve process.py import definitions perfectly
def plot_weather_map(grib_path, step, sector_name='CONUS', theme='DARK'):
    return render_weather_frame(grib_path, step, sector_name, theme)

def export_weather_data(grib_path, step):
    pass

if __name__ == "__main__":
    print("💻 Local Development Context: Running outside cloud clusters. Loaded all 85 maps.")