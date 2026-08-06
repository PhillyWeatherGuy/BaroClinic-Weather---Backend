import io
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from scipy.ndimage import gaussian_filter
from PIL import Image
from config import COLOR_CONFIGS

# ==============================================================================
# GLOBAL RENDER CONSTANTS
# ==============================================================================
FRAME_W   = 1080   # px — 3 cols × 1080 = 3240px spritesheet width (safe under 4096 WebGL limit)
FRAME_H   = 540    # px — 2:1 equirectangular ratio, perfect UV map for Three.js sphere
DPI       = 100    # locked — ensures exact pixel output every render
FIG_W     = FRAME_W / DPI   # 10.8 inches
FIG_H     = FRAME_H / DPI   # 5.4 inches

GLOBAL_EXTENT = [-180.0, 180.0, -90.0, 90.0]

# ==============================================================================
# STATIC BASE FIGURE — built once, reused every frame
# ==============================================================================
# Coastlines/borders/land drawn once at module load time.
# Each render call only redraws the temperature contour (fast).
# 50m resolution is correct for global scale — 10m is overkill and slow.

_static_fig = None
_static_ax  = None
_theme_name_cache = None

def _build_static_figure(theme_name: str):
    """
    Builds the Cartopy base figure with static geographic layers.
    Called once per theme per session. Subsequent frames reuse this canvas.
    """
    global _static_fig, _static_ax, _theme_name_cache

    theme = {
        'bg':    '#121212' if theme_name == 'DARK' else '#FFFFFF',
        'ocean': '#161920' if theme_name == 'DARK' else '#EBF2F7',
        'land':  '#1a1a1a' if theme_name == 'DARK' else '#F4F6F8',
        'lines': '#000000',
    }

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=theme['bg'])
    ax  = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent(GLOBAL_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor(theme['ocean'])

    # Geographic base layers — drawn once, never redrawn
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),  facecolor=theme['ocean'], zorder=1)
    ax.add_feature(cfeature.LAND.with_scale('50m'),   facecolor=theme['land'],  zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'),  facecolor=theme['ocean'], zorder=1.5)
    ax.coastlines(resolution='50m', color=theme['lines'], linewidth=0.4, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   edgecolor=theme['lines'], linewidth=0.3, zorder=5)

    _static_fig      = fig
    _static_ax       = ax
    _theme_name_cache = theme_name
    return fig, ax, theme


# ==============================================================================
# MAIN RENDER FUNCTION
# ==============================================================================

def render_global_frame(grib_path: str, step: int, theme_name: str = 'DARK') -> Image.Image | None:
    """
    Renders a single global equirectangular 2m Temperature frame.

    Args:
        grib_path  : Path to the ECMWF GRIB2 file for this forecast step.
        step       : Forecast hour integer (e.g. 0, 3, 6 …).
        theme_name : 'DARK' or 'LIGHT'.

    Returns:
        PIL Image (RGBA, 1080×540px) held in RAM — no disk writes.
        Returns None on failure.
    """
    global _static_fig, _static_ax, _theme_name_cache

    theme_name = theme_name.strip().upper()

    import os
    if not os.path.exists(grib_path):
        print(f"  ⚠️  GRIB file not found: {grib_path}")
        return None

    # ── 1. Build or reuse static figure ──────────────────────────────────────
    if _static_fig is None or _theme_name_cache != theme_name:
        print(f"  🗺️  Building static base figure ({theme_name} theme)…")
        fig, ax, theme = _build_static_figure(theme_name)
    else:
        fig = _static_fig
        ax  = _static_ax
        theme = {
            'bg':    '#121212' if theme_name == 'DARK' else '#FFFFFF',
            'ocean': '#161920' if theme_name == 'DARK' else '#EBF2F7',
            'land':  '#1a1a1a' if theme_name == 'DARK' else '#F4F6F8',
            'lines': '#000000',
        }

    # ── 2. Color scale setup ──────────────────────────────────────────────────
    t2m_levels = np.arange(-70.0, 130.1, 0.5)
    t2m_cfg    = COLOR_CONFIGS["TMP_2m"]
    hex_colors = (
        t2m_cfg["SHARED"]["hex_colors"]
        if "SHARED" in t2m_cfg
        else t2m_cfg[theme_name]["hex_colors"]
    )
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_t2m", hex_colors, N=len(t2m_levels) - 1
    )
    custom_norm = mcolors.BoundaryNorm(t2m_levels, ncolors=custom_cmap.N)

    # ── 3. GRIB loading & processing ─────────────────────────────────────────
    try:
        with xr.open_dataset(grib_path, engine='cfgrib',
                             backend_kwargs={'errors': 'ignore'}) as ds:
            if 'lon' in ds.coords: ds = ds.rename({'lon': 'longitude'})
            if 'lat' in ds.coords: ds = ds.rename({'lat': 'latitude'})
            ds = ds.sortby('latitude', ascending=True)
            if ds.longitude.max() > 180:
                ds = ds.assign_coords(
                    longitude=(((ds.longitude + 180) % 360) - 180)
                ).sortby('longitude')

            t2m_raw    = ds['t2m'].load()

        # K → °F with light smoothing (0.65 sigma appropriate for global scale)
        t2m_f = gaussian_filter(
            (t2m_raw.values - 273.15) * 1.8 + 32.0, sigma=0.65
        )
        lons = t2m_raw.longitude.values
        lats = t2m_raw.latitude.values

    except Exception as e:
        print(f"  ⚠️  GRIB read failed for step F{step:03d}: {e}")
        return None

    # ── 4. Remove previous dynamic artists ───────────────────────────────────
    # We only remove contourf/contour collections, not the static geo layers.
    dynamic_to_remove = [
        child for child in ax.get_children()
        if hasattr(child, 'get_paths') and child.zorder in (2, 3, 4)
    ]
    for artist in dynamic_to_remove:
        try:
            artist.remove()
        except Exception:
            pass

    # ── 5. Temperature fill ───────────────────────────────────────────────────
    ax.contourf(
        lons, lats, t2m_f,
        levels=t2m_levels,
        cmap=custom_cmap, norm=custom_norm,
        transform=ccrs.PlateCarree(),
        zorder=2, alpha=0.88, antialiased=False
    )

    # ── 6. 32°F freeze line ───────────────────────────────────────────────────
    if np.min(t2m_f) <= 32.0 <= np.max(t2m_f):
        ax.contour(
            lons, lats, t2m_f,
            levels=[32.0], colors=['#4169E1'], linewidths=[0.7],
            transform=ccrs.PlateCarree(), zorder=4
        )

    # ── 7. Capture frame to PIL Image via in-memory buffer ───────────────────
    # No disk I/O — frame lives in RAM until spritesheet stacking.
    buf = io.BytesIO()
    fig.savefig(
        buf, format='png', dpi=DPI,
        facecolor=theme['bg'], edgecolor='none',
        bbox_inches=None, pad_inches=0
    )
    buf.seek(0)
    pil_frame = Image.open(buf).copy().convert('RGBA')
    buf.close()

    return pil_frame


# ==============================================================================
# CLEANUP HELPER — call at end of a full run to free matplotlib memory
# ==============================================================================

def close_static_figure():
    """Closes the shared static figure. Call once after all frames are rendered."""
    global _static_fig, _static_ax, _theme_name_cache
    if _static_fig is not None:
        plt.close(_static_fig)
        _static_fig      = None
        _static_ax       = None
        _theme_name_cache = None
        print("  🧹  Static base figure closed and memory freed.")


# ==============================================================================
# LEGACY ADAPTER — preserves process.py import signature exactly
# ==============================================================================

def plot_weather_map(grib_path: str, step: int, sector_name: str = 'GLOBAL', theme: str = 'DARK'):
    """
    Drop-in replacement for the old sector-based plot_weather_map.
    sector_name is accepted but ignored — all renders are now global.
    Returns the PIL Image frame (process.py handles stacking + saving).
    """
    return render_global_frame(grib_path, step, theme_name=theme)

def export_weather_data(grib_path: str, step: int):
    """Retained stub for process.py import compatibility."""
    pass


if __name__ == "__main__":
    print("💻 plot_maps.py — Global Equirectangular Renderer loaded.")
    print(f"   Output: {FRAME_W}×{FRAME_H}px | DPI: {DPI} | Theme: DARK/LIGHT")
