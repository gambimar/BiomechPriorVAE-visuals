#!/usr/bin/env python3
"""
Generate consolidated Fukuchi mean/var files (25, 35, 45 m/s) from raw tracking data
with proper ankle angle inclusion.
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path

def extract_variables_from_fukuchi_raw(mat_data):
    """Extract hip, knee, ankle, Fx, Fy from raw Fukuchi dataStruct."""
    
    # Navigate the nested structure
    if 'dataStruct' not in mat_data:
        return None
    
    ds = mat_data['dataStruct']
    if isinstance(ds, np.ndarray):
        if ds.shape[0] > 0:
            ds = ds[0, 0]  # Get the first element of 1x1 array
    
    if not hasattr(ds, 'dtype') or 'variables' not in ds.dtype.names:
        return None
    
    variables = ds['variables']
    if isinstance(variables, np.ndarray) and variables.size == 1:
        variables = variables[0]
    
    result = {}
    
    # Iterate through variables
    try:
        for i in range(len(variables)):
            var_obj = variables[i]
            
            # Extract name
            name = ''
            if hasattr(var_obj, 'name'):
                name = var_obj.name
            elif 'name' in var_obj.dtype.names if isinstance(var_obj, np.ndarray) else False:
                name = var_obj['name']
            
            if isinstance(name, np.ndarray):
                name = name[0] if name.size > 0 else ''
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            name = str(name).lower()
            
            # Extract data (avg or mean)
            values = None
            for field_name in ['avg', 'mean']:
                if hasattr(var_obj, field_name):
                    values = getattr(var_obj, field_name)
                    break
                elif isinstance(var_obj, np.ndarray) and field_name in var_obj.dtype.names:
                    values = var_obj[field_name]
                    break
            
            if values is None:
                continue
            
            # Convert to 1D if needed
            if isinstance(values, np.ndarray):
                values = values.flatten()
            else:
                values = np.array(values).flatten()
            
            # Categorize by name
            if 'hip' in name and ('flexion' in name or 'angle' in name):
                result['hip'] = values
            elif 'knee' in name and ('flexion' in name or 'angle' in name):
                result['knee'] = values
            elif 'ankle' in name and ('flexion' in name or 'dorsiflexion' in name or 'angle' in name):
                result['ankle'] = values
            elif 'grf' in name or 'ground' in name:
                if 'fore' in name or 'x' in name:
                    result['Fx'] = values
                elif 'vertical' in name or 'y' in name:
                    result['Fy'] = values
    
    except Exception as e:
        print(f"  Warning: Error extracting variables: {e}")
    
    return result if result else None


def generate_consolidated_files():
    """Generate Fukuchi_25/35/45_mean/var.mat files from raw tracking data."""
    
    src_dir = 'reference_data/fukuchi/2017_Fukuchi_TrackingDataFormat'
    dst_dir = 'reference_data'
    
    speeds = {'T25': 2.5, 'T35': 3.5, 'T45': 4.5}
    
    for suffix, speed in speeds.items():
        print(f"\nProcessing {suffix} (speed={speed} m/s)...")
        
        # Find all files for this speed
        src_path = Path(src_dir)
        files = sorted(src_path.glob(f'*{suffix}.mat'))
        
        if not files:
            print(f"  No files found for {suffix}")
            continue
        
        print(f"  Found {len(files)} subjects")
        
        # Storage for all subjects
        all_data = {'hip': [], 'knee': [], 'ankle': [], 'Fx': [], 'Fy': []}
        
        for file_path in files:
            try:
                # Load raw data
                raw = loadmat(str(file_path), squeeze_me=False, simplify_cells=True)
                
                # Extract variables
                var_data = extract_variables_from_fukuchi_raw(raw)
                
                if var_data is None:
                    print(f"  Skipping {file_path.name}: could not extract variables")
                    continue
                
                # Verify we have angles (hip, knee at minimum)
                if 'hip' not in var_data or 'knee' not in var_data:
                    print(f"  Skipping {file_path.name}: missing hip or knee data")
                    continue
                
                # Store data
                for key in ['hip', 'knee', 'ankle', 'Fx', 'Fy']:
                    if key in var_data and len(var_data[key]) > 0:
                        all_data[key].append(var_data[key])
                
                print(f"  OK: {file_path.name}")
            
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        # Compute means and variances
        if all_data['hip'] and all_data['knee']:
            print(f"  Computed statistics from {len(all_data['hip'])} subjects")
            
            # Convert to 2D arrays and compute statistics
            hip_array = np.array(all_data['hip'])
            knee_array = np.array(all_data['knee'])
            ankle_array = np.array(all_data['ankle']) if all_data['ankle'] else np.zeros_like(hip_array)
            
            # Ensure 100-sample norm (interpolate if needed)
            if hip_array.shape[1] != 100:
                print(f"  Warning: Data has {hip_array.shape[1]} samples, normalizing to 100")
                x_old = np.linspace(0, 1, hip_array.shape[1])
                x_new = np.linspace(0, 1, 100)
                hip_array = np.array([np.interp(x_new, x_old, row) for row in hip_array])
                knee_array = np.array([np.interp(x_new, x_old, row) for row in knee_array])
                ankle_array = np.array([np.interp(x_new, x_old, row) for row in ankle_array])
            
            # Compute mean and variance
            mean_hip = np.mean(hip_array, axis=0)
            var_hip = np.var(hip_array, axis=0, ddof=1)  # Sample variance
            mean_knee = np.mean(knee_array, axis=0)
            var_knee = np.var(knee_array, axis=0, ddof=1)
            mean_ankle = np.mean(ankle_array, axis=0)
            var_ankle = np.var(ankle_array, axis=0, ddof=1)
            
            # Stack into 3x100 arrays
            arr_mean = np.array([mean_hip, mean_knee, mean_ankle])
            arr_var = np.array([var_hip, var_knee, var_ankle])
            
            # Save
            speed_int = int(speed * 10)
            mean_file = os.path.join(dst_dir, f'Fukuchi_{speed_int}_mean.mat')
            var_file = os.path.join(dst_dir, f'Fukuchi_{speed_int}_var.mat')
            
            savemat(mean_file, {'arr': arr_mean}, format='4')
            savemat(var_file, {'arr': arr_var}, format='4')
            
            print(f"  Saved: {os.path.basename(mean_file)}, {os.path.basename(var_file)}")
            print(f"  Shape: {arr_mean.shape} (hip, knee, ankle x gait cycle samples)")
            print(f"  Sample means - hip[0]={mean_hip[0]:.4f}, knee[0]={mean_knee[0]:.4f}, ankle[0]={mean_ankle[0]:.4f}")
        else:
            print(f"  No valid data found for {suffix}")


if __name__ == '__main__':
    os.chdir('/Users/markusgambietz/PhD/01_Python_Projects/biomechpriorVAE')
    generate_consolidated_files()
    print("\nConsolidation complete!")
