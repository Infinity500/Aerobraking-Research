# Imports
import numpy as np
import math
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import subprocess
import time

# Configuration Variables - Initial State of Spacecraft
CONFIG = {
    # MarsGRAM setup done locally. Replace with local paths for MarsGRAM outputs.
    'marsgram_exe_path': r"C:\Users\savit\OneDrive\Documents\AerospaceResearch\Code\GRAM_Suite_2.1.0\Windows\MarsGRAM.exe",
    'marsgram_csv_initial': r"C:\Users\savit\OneDrive\Documents\AerospaceResearch\Code\GRAM_Suite_2.1.0\Outputs\myref_OUTPUT.csv",
    'ref_input_template_path': r"C:\Users\savit\Earth-GRAM\ref_input_generated.txt",
    'traj_file_path': r"C:\Users\savit\Earth-GRAM\TRAJDATA_allpasses.txt",

    'retro_dv_magnitude': 0,    # Retrograde delta-v (in m/s); Use 0 for no retro-propulsion
    'lat0': 40.0,               # Injection Latitude (in degrees)
    'lon0': 60.0,               # Injection Longitude (in degrees)
    'alt0_km': 130,             # Injection Altitude (in km)
    'speed': 4800.0,            # Injection velocity (in m/s)
    'fpa': -8,                  # Injection Flight Path Angle (in degrees); normally between ~7.5 - 8.5 degrees
    'az': 90.0,                 # Azimuth (Used to set velocity vector)
    'G_REF': 3.721,             # Mars gravity reference (in m/s^2)
    'target_perigee_km': 2.0,   # Final desired perigee alittude (in km)
    'max_passes': 200,          # Maximum passes through orbit before termination due to computational limitations
    'dt': 0.5,                  # Propogation delta-time (in seconds)
    'g_limit': 7.0,             # Abort with peak g-load (Force)
    'q_limit': 6e4,             # Abort with peak q-load (Dynamic Pressure)
    'burn_location': 'apoapsis' # Location for retrograde delta-v
}

#Constants
MU_MARS = 4.28284e13                 # m^3/s^2
R_MARS  = 3389.5e3                   # meters
OMEGA_MARS = 2.0 * math.pi / 88642.0 # Angular velocity (rad/s) sidereal
MASS = 8500.0                        # Orion (in kg)
RADIUS = 2.51                        # Orion (radius of heat shield) in meters
AREA = math.pi * RADIUS**2           # m^2
CD = 2.0                             # Blunt-body coefficent of drag            
pass_results = []

# ---Basic Functions (1)---
def enu_basis(lat_rad, lon_rad):
    # Function 1.1: Used to return directional unit vectors from latitude and longitude.
    sinlat = math.sin(lat_rad)
    coslat = math.cos(lat_rad)
    sinlon = math.sin(lon_rad)
    coslon = math.cos(lon_rad)
    up = np.array([coslat * coslon, coslat * sinlon, sinlat])
    north = np.array([-sinlat * coslon, -sinlat * sinlon, coslat])
    north /= np.linalg.norm(north)
    east = np.array([-sinlon, coslon, 0.0])
    east /= np.linalg.norm(east)
    return east, north, up

def build_initial_state(lat_deg, lon_deg, alt_m, speed_m_s, fpa_deg, azimuth_deg):
    # Function 1.2: Converts into 3D position+velocity vectors in Mars-centered frame.
    # Assumes that spacecraft velocity is purely intertial (like incoming from interplanetary transfer) 
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    rmag = R_MARS + alt_m
    x = rmag * math.cos(lat) * math.cos(lon)
    y = rmag * math.cos(lat) * math.sin(lon)
    z = rmag * math.sin(lat)
    r0 = np.array([x, y, z])
    east, north, up = enu_basis(lat, lon)
    az = math.radians(azimuth_deg)
    horiz_dir = math.cos(az) * north + math.sin(az) * east
    gamma = math.radians(fpa_deg)
    v0 = speed_m_s * (math.cos(gamma) * horiz_dir + math.sin(gamma) * up)
    return r0, v0

def cartesian_to_geo(r):
    # Function 1.3: Converts from Cartesian (Mars-centered) to Geographical coordinates (Lat, Long, Alt).
    x, y, z = r
    rnorm = np.linalg.norm(r)
    lat = math.asin(z / rnorm)
    lon = math.atan2(y, x)
    h = rnorm - R_MARS
    return math.degrees(lat), math.degrees(lon), h

def geo_to_cartesian(lat, lon, alt):
    # Function 1.4: Reverts back Geographical coordinates to Cartesian.
    lat = np.radians(lat)
    lon = np.radians(lon)
    r = R_MARS + alt * 1000.0
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

def orbital_elements(r, v):
    # Function 1.5: Computates orbitial elemants from radius and velocity vectors using Keplerian orbital formulas.
    # Assumes Mars as point mass.
    rvec = np.array(r)
    vvec = np.array(v)
    rmag = np.linalg.norm(rvec)
    vmag = np.linalg.norm(vvec)
    hvec = np.cross(rvec, vvec)
    hmag = np.linalg.norm(hvec)
    evec = (np.cross(vvec, hvec) / MU_MARS) - (rvec / rmag)
    emag = np.linalg.norm(evec)
    energy = 0.5 * vmag*vmag - MU_MARS / rmag
    if abs(energy) < 1e-12:
        a = np.inf
    else:
        a = - MU_MARS / (2.0 * energy)
    rp = a * (1.0 - emag) if np.isfinite(a) else np.nan
    ra = a * (1.0 + emag) if np.isfinite(a) else np.nan
    return {'a_m': a, 'e': emag, 'rp_m': rp, 'ra_m': ra, 'h_vec': hvec, 'e_vec': evec}

def read_marsgram_csv_robust(csv_path, rho_clamp_max=0.05, rho_floor=1e-14, n_fit=8):
    # Function 1.6: Reads MarsGram-outputted CSV with extrapolation methods to discern atmospheric densities
    df = pd.read_csv(csv_path)
    cols_low = [c.lower() for c in df.columns]
    
    alt_col = None
    for cand in ['height_km','height(km)','height','alt_km','altitude_km','alt']:
        if cand in cols_low:
            alt_col = df.columns[cols_low.index(cand)]
            break
    if alt_col is None:
        raise RuntimeError("[ERROR]: Cannot find altitude column in MarsGRAM CSV.")
    
    rho_col = None
    for cand in ['density_kgm3','density(kg/m3)','density','rho','density_kg/m3']:
        if cand in cols_low:
            rho_col = df.columns[cols_low.index(cand)]
            break

    if rho_col is None:
        raise RuntimeError("[ERROR]: Cannot find density column in MarsGRAM CSV.")

    alt_km = df[alt_col].astype(float).values
    rho_vals = df[rho_col].astype(float).values
    
    unique_alts, inv = np.unique(alt_km, return_inverse=True)
    rho_mean = np.zeros_like(unique_alts, dtype=float)
    for i,ua in enumerate(unique_alts):
        rho_mean[i] = np.mean(rho_vals[inv==i])
    order = np.argsort(unique_alts)
    alt_km_sorted = unique_alts[order]
    rho_sorted = rho_mean[order]
    interp = interp1d(alt_km_sorted, rho_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    fit_n = min(n_fit, len(alt_km_sorted))
    bottom_alts = alt_km_sorted[:fit_n]
    bottom_rhos = rho_sorted[:fit_n]
    mask = bottom_rhos > 0
    if mask.sum() >= 2:
        x = bottom_alts[mask]
        y = np.log(bottom_rhos[mask])
        B, A = np.polyfit(x, y, 1)   
        def rho_extrap_down(km):
            return float(np.exp(A + B * km))
    else:
        min_rho = float(rho_sorted[0])
        def rho_extrap_down(km):
            return float(min_rho)
    alt_min_km = float(alt_km_sorted[0])
    alt_max_km = float(alt_km_sorted[-1])
    def rho_of_h_m(h_m):
        km = float(h_m) / 1000.0
        if km < alt_min_km:
            val = rho_extrap_down(km)
        elif km > alt_max_km:
            val = float(interp(alt_max_km)) * math.exp(-(km-alt_max_km)/10.0)
        else:
            val = float(interp(km))
        if np.isnan(val):
            val = rho_floor
        val = max(rho_floor, min(val, rho_clamp_max))
        return float(val)
    rho_of_h_m._alt_min_km = alt_min_km
    rho_of_h_m._alt_max_km = alt_max_km
    rho_of_h_m._sampled_alts = alt_km_sorted
    rho_of_h_m._sampled_rhos = rho_sorted
    print(f"[INFO] Built rho(h) from '{csv_path}' (alts {alt_min_km:.2f}-{alt_max_km:.2f} km, clamp {rho_clamp_max}).")
    return rho_of_h_m

def check_initial_orbit(lat, lon, alt_km, speed, fpa, az):
    # Function 1.7: Uses state vectors to check if orbit is valid/safe
    r0, v0 = build_initial_state(lat, lon, alt_km*1000, speed, fpa, az)
    elems = orbital_elements(r0, v0)
    rp_km = (elems['rp_m'] - R_MARS) / 1000.0
    print(f"[CHECK] Initial orbit periapsis altitude: {rp_km:.2f} km")
    if rp_km <= 0:
        print("[WARN] Initial periapsis is below Mars surface! Adjust speed or flight path angle.")
    else:
        print("[CHECK] Initial orbit periapsis is safe.")
    return r0, v0, rp_km

# Propogator Functions (2)
def drag_acceleration(r, v, rho_func, cd=CD, area=AREA, mass=MASS):
    # Function 2.1: Calculates drag using state vectors
    # Assumes identical velocity for surface and atmosphere (First-order model so ignores winds); ignores hypersonic effects; and no lift or pitching accounted
    omega_vec = np.array([0.0, 0.0, OMEGA_MARS]) # Converts angular velocity to a vector
    v_atm = np.cross(omega_vec, r)               # Cross product between omega and r to get velocity of surface
    v_rel = v - v_atm                            # Relative speed of spacecraft to surface
    vrel_norm = np.linalg.norm(v_rel)            # Normalize vector to only get direction
    if vrel_norm <= 1e-9:
        return np.zeros(3), 0.0, 0.0
    h = np.linalg.norm(r) - R_MARS
    rho = rho_func(max(h, 0.0))
    q = 0.5 * rho * vrel_norm**2                 # Get dynamic pressure
    f_D = q * cd * area                          # Get force of drag
    a_drag_mag = f_D / mass                      # Get magnitude acceleration of drag
    a_drag = - a_drag_mag * (v_rel / vrel_norm)  # Convert this to a vector to get vectored acceleration of drag
    return a_drag, a_drag_mag, q

def dynamics(t, y, rho_func):
    # Function 2.2: Gets time derivative from state vectors (radius + velocity) with drag and gravity
    r = y[:3]                                    # Gets first three elemants in radius vector (x,y,z)
    v = y[3:]                                    # Gets first three elemants in velocity vector (vx, vy, vz)
    rnorm = np.linalg.norm(r)                    # Gets total distance from Mars' center
    a_drag, _, _ = drag_acceleration(r, v, rho_func) # Gets acceleration due to drag
    a_grav = -MU_MARS * r / (rnorm**3)           # Gets acceleration due to gravity
    a_total = a_drag + a_grav
    return np.concatenate([v, a_total])

def rk4_step(t, y, dt, rho_func):
    # Function 2.3: Advances state vectors every [dt] step using 4th order Runge-Kutta method
    # Used rather than a simple Euler's method for higher acccuracy 
    k1 = dynamics(t, y, rho_func)
    k2 = dynamics(t + dt/2, y + dt * k1 / 2, rho_func)
    k3 = dynamics(t + dt/2, y + dt * k2 / 2, rho_func)
    k4 = dynamics(t + dt, y + dt * k3, rho_func)
    y_next = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next


def propagate_single_pass(r0, v0, rho_func, dt=0.1, atm_entry_h=66e3, atm_exit_h=66e3, max_steps=1000000):
    # Function 2.4: Uses other helper function to simulate one aerbraking pass through atmosphere
    times = [0.0]
    rs = [r0.copy()]
    vs = [v0.copy()]
    t = 0.0
    entered = False
    step = 0
    y = np.concatenate([r0.copy(), v0.copy()])
    while step < max_steps:
        r = y[:3]
        v = y[3:]
        rnorm = np.linalg.norm(r)
        h = rnorm - R_MARS
        if rnorm <= R_MARS:                      # Breaks program if spacecraft crashes to surface
            print("[WARN] Planetary impact detected (r <= R_MARS). Terminating pass.")
            break
        y = rk4_step(t, y, dt, rho_func)         # Uses Rk4 propogation to get spacecraft state
        t += dt
        times.append(t)
        rs.append(y[:3].copy())
        vs.append(y[3:].copy())
        if step % 10000 == 0:
            print(f"[DEBUG] Step {step}: t={t:.1f}s, Altitude={h/1000:.2f} km, Entered={entered}")
        if (not entered) and (h < atm_entry_h):
            entered = True
            print(f"[INFO] Atmosphere entry detected at t={t:.1f}s, altitude={h/1000:.2f} km")
        if entered and (h > atm_exit_h) and (t > 1.0):
            print(f"[INFO] Atmosphere exit detected at t={t:.1f}s, altitude={h/1000:.2f} km")
            break
        step += 1
    if step >= max_steps:
        print(f"[WARN] Maximum steps ({max_steps}) reached without atmosphere exit; terminating pass.")
    return np.array(times), np.array(rs), np.array(vs)


def analyze_pass(times, rs, vs, rho_func, burn_loc=CONFIG['burn_location'], G_REF=CONFIG['G_REF']):
    # Function 2.5: Uses previous data from aerobraking pass to find key metrics (peak deceleration, total velocity loss, and peak dynamic pressure)
    N = len(times)
    decels = np.zeros(N)                         # Array for storing deceleration
    qdyn = np.zeros(N)                           # Array for storing dynamic pressure
    dt_array = np.diff(times, prepend=times[0])  # Array for delta-time
    for i in range(N):
        __, a_drag_mag, q = drag_acceleration(rs[i], vs[i], rho_func) # Gets force of drag to compute deceleration and dynamic pressure
        decels[i] = a_drag_mag
        qdyn[i] = q
    delta_v = 0.0
    for i in range(1, N):   # Gets total velocity lost due to drag
        a_mid = 0.5*(decels[i] + decels[i-1])
        delta_v += a_mid * dt_array[i]
    peak_g = np.max(decels)/G_REF
    peak_q = np.max(qdyn)
    altitudes = np.linalg.norm(rs, axis=1) - R_MARS
    if burn_loc == 'periapsis':
        alt_idx = np.argmin(altitudes)
    elif burn_loc == 'apoapsis':
        alt_idx = np.argmax(altitudes)
    return {'peak_g': peak_g, 'delta_v_m_s': delta_v, 'peak_q_Pa': peak_q, 'decels': decels, 'qdyn': qdyn, 'max_alt_idx': alt_idx}

def write_trajdata_all(t_global, r_global, filename_all='TRAJDATA_allpasses.txt', filename_last='TRAJDATA_lastpass.txt'):
    # Function 2.6: Uses all orbital data to create trajectory files of spacecraft during aerobraking.
    with open(filename_all, 'w') as f:
        for t, r in zip(t_global, r_global):
            lat, lon, h = cartesian_to_geo(r)
            f.write(f"{t:12.3f}  {h/1000.0:10.6f}  {lat:10.6f}  {lon:10.6f}\n")
    print(f"[INFO] Wrote {len(t_global)} points to '{filename_all}'")
    L = max(1, int(len(t_global)/10))
    with open(filename_last, 'w') as f:
        for t, r in zip(t_global[-L:], r_global[-L:]):
            lat, lon, h = cartesian_to_geo(r)
            f.write(f"{t:12.3f}  {h/1000.0:10.6f}  {lat:10.6f}  {lon:10.6f}\n")
    print(f"[INFO] Wrote last-pass snippet ({L} points) to '{filename_last}'")

def write_ref_input_template(traj_filename='TRAJDATA_allpasses.txt', out_name='ref_input_generated.txt', data_filename=r'C:\Users\savit\OneDrive\Documents\AerospaceResearch\Code\GRAM_Suite_2.1.0\Mars\data', spice_filename=r'C:\Users\savit\OneDrive\Documents\AerospaceResearch\Code\spice'):
    # Function 2.7: Uses the trajectory file to create a ref_input file for MarsGRAM
    content = f"""$INPUT
 DataPath              = '{data_filename}'
 SpicePath             = {spice_filename}
 TrajectoryFileName    = '{traj_filename}'
 EastLongitudePositive = 1
 IsPlanetoCentric      = 1
 Month                 = 01
 Day                   = 01
 Year                  = 2035
 Hour                  = 00
 Minute                = 00
 Seconds               = 0.0
 F107                  = 68.0
 MGCMConstantDustLevel = 0.3
 $END
"""
    with open(out_name, 'w') as f:
        f.write(content)
    print(f"[INFO] ref_input template written to '{out_name}' (edit paths for your system)")

def run_marsgram(ref_input_path, marsgram_exe_path):
    # Function 2.8: Reruns MarsGRAM after receiving Trajectory_ALL file.
    print(f"[INFO] Running MarsGRAM with input file: {ref_input_path}")
    try:
        result = subprocess.run([marsgram_exe_path, ref_input_path], capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print("[ERROR] MarsGRAM failed:")
            print(result.stderr)
            return False
        return True
    except Exception as e:
        print(f"[ERROR] MarsGRAM execution exception: {e}")
        return False

# Visualization Functions (3)
def plot_altitude_vs_time(times, positions):
    # Function 3.1: Graphs altitude vs time for entire duration of aerobraking
    altitudes_km = [(np.linalg.norm(r) - R_MARS)/1000.0 for r in positions]
    plt.figure(figsize=(10,6))
    plt.plot(np.array(times)/60.0, altitudes_km, label='Altitude (km)')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (km)')
    plt.title('Altitude vs Time Over Passes')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_trajectory(traj_all_file):
    # Function 3.2: Uses trajectory file to create 3D model of spacecraft orbital paths
    times, alts, lats, lons = [], [], [], []
    with open(traj_all_file, 'r') as f:
        for line in f:
            t, h, lat, lon = line.split()
            times.append(float(t))
            alts.append(float(h))
            lats.append(float(lat))
            lons.append(float(lon))
    xs, ys, zs = [], [], []
    for lat, lon, alt in zip(lats, lons, alts):
        x, y, z = geo_to_cartesian(lat, lon, alt)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)   
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = R_MARS * np.outer(np.cos(u), np.sin(v))
    y_sphere = R_MARS * np.outer(np.sin(u), np.sin(v))
    z_sphere = R_MARS * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='orange', alpha=0.5, zorder=0)
    ax.plot(xs, ys, zs, color='blue', label='Trajectory', linewidth=1)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Aerobraking Trajectory around Mars')
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend()
    plt.show()

def multi_pass_aerobrake(cfg):
    # Main function that combines all others to run full multi-pass aerobraking simulation
    lat0 = cfg['lat0']
    lon0 = cfg['lon0']
    alt0_km = cfg['alt0_km']
    speed = cfg['speed']
    fpa = cfg['fpa']
    az = cfg['az']
    target_perigee_km = cfg['target_perigee_km']
    max_passes = cfg['max_passes']
    dt = cfg['dt']
    g_limit = cfg['g_limit']
    q_limit = cfg['q_limit']
    marsgram_exe_path = cfg['marsgram_exe_path']
    marsgram_csv_path = cfg['marsgram_csv_initial']
    ref_input_path = cfg['ref_input_template_path']
    traj_file_path = cfg['traj_file_path']
    print("[START] Checking initial orbit conditions...")
    r_current, v_current, rp0 = check_initial_orbit(lat0, lon0, alt0_km, speed, fpa, az)    # Runs initial safety checks
    if rp0 <= 0:
        print("[ABORT] Initial orbit periapsis below surface. Adjust inputs and rerun.")
        return
    print(f"[LOAD] Loading MarsGRAM atmosphere from '{marsgram_csv_path}'")
    rho_func = read_marsgram_csv_robust(marsgram_csv_path)
    all_times = []
    all_positions = []
    t_global = 0.0
    for pass_num in range(1, max_passes+1):
        print(f"\n[PASS {pass_num}] Starting pass with initial altitude {(np.linalg.norm(r_current)-R_MARS)/1000:.2f} km and initial velocity of {(np.linalg.norm(v_current)):.2f} m/s")
        times, rs, vs = propagate_single_pass(r_current, v_current, rho_func, dt=dt)    # Propogates passes using function
        times = times + t_global
        t_global = times[-1]
        all_times.extend(times)
        all_positions.extend(rs)        # Accumulates times into list
        results = analyze_pass(times, rs, vs, rho_func)
        retro_dv_mag = cfg.get('retro_dv_magnitude', 0.0)
        retro_dv_mag = cfg.get('retro_dv_magnitude', 0.0)
        if retro_dv_mag > 0.0:                  # Applies retrograde delta-v if applicable
            idx = results['max_alt_idx']
            r_burn = rs[int(idx)]
            v_burn = vs[idx]
            v_norm = np.linalg.norm(v_burn)
            retro_dir = -v_burn / v_norm
            v_delta = retro_dv_mag * retro_dir 
            print(f"[BURN] Retrograde burn of {retro_dv_mag:.1f} m/s applied at altitude {(np.linalg.norm(r_burn)-R_MARS)/1000:.2f} km")
            r_current = r_burn
            v_current = v_burn + v_delta
        else:
            r_current = rs[-1]
            v_current = vs[-1]
        peak_g = results['peak_g']
        delta_v = results['delta_v_m_s']
        peak_q = results['peak_q_Pa']
        print(f"[RESULTS] Pass {pass_num}: peak g-load = {peak_g:.2f} g, delta-v removed = {delta_v:.1f} m/s, peak q = {peak_q/1000:.1f} kPa")
        pass_results.append({
            'Pass': pass_num,
            'Peak G-Load (g)': round(peak_g, 2),
            'Delta-V Removed (m/s)': round(delta_v, 1),
            'Peak Q (kPa)': round(peak_q / 1000, 1)
        })
        if peak_g > g_limit:
            print(f"[ABORT] Exceeded g-load limit ({peak_g:.2f} g > {g_limit} g)")
            break
        if peak_q > q_limit:
            print(f"[ABORT] Exceeded dynamic pressure limit ({peak_q/1000:.1f} kPa > {q_limit/1000} kPa)")
            break
        elems = orbital_elements(r_current, v_current)
        rp_km = (elems['rp_m'] - R_MARS)/1000.0
        print(f"[ORBIT] New periapsis altitude after pass {pass_num}: {rp_km:.3f} km")
        if rp_km <= target_perigee_km:
            v_final = math.sqrt((v_current[0]**2)+(v_current[1]**2)+(v_current[2]**2))
            print(f"[SUCCESS] Target periapsis {target_perigee_km} km reached after pass {pass_num}. Final velocity was {v_final:.2f} m/s.")
            break
        write_trajdata_all(np.array(all_times), np.array(all_positions), filename_all='TRAJDATA_allpasses.txt')
        write_ref_input_template(traj_filename='TRAJDATA_allpasses.txt', out_name=ref_input_path)        
        success = run_marsgram(ref_input_path, marsgram_exe_path)
        if not success:
            print("[ABORT] MarsGRAM failed; aborting aerobraking loop.")
            break
        time.sleep(2)  
        rho_func = read_marsgram_csv_robust(marsgram_csv_path)
    write_trajdata_all(np.array(all_times), np.array(all_positions))
    plot_altitude_vs_time(np.array(all_times), np.array(all_positions))
    draw_trajectory(traj_file_path)
    all_times = np.array(all_times)
    all_positions = np.array(all_positions)
    peak_gs = [res['Peak G-Load (g)'] for res in pass_results]
    plt.figure(figsize=(8,5)) # Creates plot for Peak-G over passes
    plt.plot(range(1, len(peak_gs)+1), peak_gs, marker='o', color='red')
    plt.xlabel('Pass Number')
    plt.ylabel('Peak G [g]')
    plt.title('Peak G-Load per Pass')
    plt.grid(True)
    plt.show()
    delta_vs = [res['Delta-V Removed (m/s)'] for res in pass_results]
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(delta_vs)+1), delta_vs, marker='x', color='green')
    plt.xlabel('Pass Number') # Creates plot for Delta-V over passes
    plt.ylabel('Delta-V Removed [m/s]')
    plt.title('Delta-V Removed per Pass')
    plt.grid(True)
    plt.show()
    total_time_min = all_times[-1]/60.0
    print(f"[INFO] Total simulation time: {total_time_min:.2f} minutes ({all_times[-1]:.1f} seconds)")
    print(f"[INFO] Number of passes simulated: {len(pass_results)}")
    print("[DONE] Multi-pass aerobraking simulation completed. Drawing trajectory around Mars.")

if __name__ == "__main__":
    multi_pass_aerobrake(CONFIG)
    df = pd.DataFrame(pass_results)
    df.to_csv('aerobrake_results.csv', index=False)

"""
Overall, this simulation is able to:
1) Get initial states
2) Interpolate atmospheric densities
3) Integrate using RK4
4) Analyze aerodynamic effects
5) Apply retrograde delta-v

Sources for programming:
- https://ntrs.nasa.gov/api/citations/20210024320/downloads/MEADSreconstruction.pdf 
- https://doi.org/10.1016/j.actaastro.2023.11.045
- https://doi.org/10.2514/3.25554
- https://medium.com/@viscircuit/euler-vs-runge-kutta-methods-de34565bf0cf

  Libraries and Tools:
  - Numpy
  - Matplotlib
  - MarsGRAM (Mars Global Reference Atmospheric Model)
  - Mars Climate Database 
  - Pandas
  - SciPy
"""

