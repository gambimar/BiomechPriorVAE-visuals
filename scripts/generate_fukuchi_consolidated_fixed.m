function generate_fukuchi_consolidated_fixed()
    % generate_fukuchi_consolidated_fixed: Create mean/var files for all Fukuchi speeds (2.5, 3.5, 4.5)
    % FIX: vars is a table, use vars(i, :) not vars(i)
    
    fprintf('[DEBUG] STARTING generate_fukuchi_consolidated_fixed\n');
    
    srcDir = 'reference_data/fukuchi/2017_Fukuchi_TrackingDataFormat';
    dstDir = 'reference_data';
    
    speeds = struct('T25', 2.5, 'T35', 3.5, 'T45', 4.5);
    
    for speed_suffix = {'T25', 'T35', 'T45'}
        suffix = speed_suffix{1};
        speed = speeds.(suffix);
        
        fprintf('\n[DEBUG] ========== SPEED: %s (%.1f m/s) ==========\n', suffix, speed);
        
        % Find all files for this speed
        pattern = fullfile(srcDir, ['*' suffix '.mat']);
        d = dir(pattern);
        files = {d.name};
        
        fprintf('[DEBUG] Found %d files\n', numel(files));
        
        if isempty(files)
            continue;
        end
        
        % Initialize storage
        all_hip = [];
        all_knee = [];
        all_ankle = [];
        success_count = 0;
        
        % Process each subject
        for i = 1:numel(files)
            fprintf('[DEBUG] [%d/%d] %s... ', i, numel(files), files{i});
            
            inFile = fullfile(srcDir, files{i});
            
            try
                data = load(inFile);
                
                if ~isfield(data, 'dataStruct')
                    fprintf('SKIP (no dataStruct)\n');
                    continue;
                end
                
                ds = data.dataStruct(1);
                
                if ~isfield(ds, 'variables')
                    fprintf('SKIP (no variables)\n');
                    continue;
                end
                
                vars = ds.variables;
                
                % Extract joints
                hip_data = [];
                knee_data = [];
                ankle_data = [];
                
                % IMPORTANT: vars is a TABLE, so use (i, :) indexing
                for v_idx = 1:height(vars)
                    var_row = vars(v_idx, :);  % FIX: Use (v_idx, :) not (v_idx)
                    
                    % Convert table row to struct
                    var_struct = table2struct(var_row);
                    if isstruct(var_struct)
                        var_obj = var_struct;
                    else
                        continue;
                    end
                    
                    % Get name
                    name = '';
                    if isfield(var_obj, 'name')
                        name_val = var_obj.name;
                        if ischar(name_val)
                            name = name_val;
                        elseif iscellstr(name_val) || iscell(name_val)
                            name = char(name_val);
                        end
                    end
                    
                    if isempty(name)
                        continue;
                    end
                    
                    % Get values
                    values = [];
                    if isfield(var_obj, 'avg')
                        values = var_obj.avg;
                    elseif isfield(var_obj, 'mean')
                        values = var_obj.mean;
                    end
                    
                    if isempty(values) || ~isnumeric(values)
                        continue;
                    end
                    
                    % Make sure values is a row vector
                    if iscolumn(values)
                        values = values';
                    end
                    
                    % Categorize
                    name_lower = lower(name);
                    if contains(name_lower, 'hip') && (contains(name_lower, 'flexion') || contains(name_lower, 'angle'))
                        hip_data = values;
                    elseif contains(name_lower, 'knee') && (contains(name_lower, 'flexion') || contains(name_lower, 'angle'))
                        knee_data = values;
                    elseif contains(name_lower, 'ankle') && (contains(name_lower, 'flexion') || contains(name_lower, 'dorsiflexion') || contains(name_lower, 'angle'))
                        ankle_data = values;
                    end
                end
                
                % Store if complete
                if ~isempty(hip_data) && ~isempty(knee_data) && ~isempty(ankle_data)
                    all_hip = [all_hip; hip_data];
                    all_knee = [all_knee; knee_data];
                    all_ankle = [all_ankle; ankle_data];
                    success_count = success_count + 1;
                    fprintf('OK\n');
                else
                    fprintf('SKIP (incomplete)\n');
                end
                
            catch e
                fprintf('ERROR: %s\n', e.message);
            end
        end
        
        fprintf('[DEBUG] Aggregation: %d/%d successful\n', success_count, numel(files));
        
        % Compute statistics and save
        if ~isempty(all_hip)
            mean_hip = mean(all_hip, 1);
            var_hip = var(all_hip, 0, 1);
            mean_knee = mean(all_knee, 1);
            var_knee = var(all_knee, 0, 1);
            mean_ankle = mean(all_ankle, 1);
            var_ankle = var(all_ankle, 0, 1);
            
            arr = [mean_hip; mean_knee; mean_ankle];
            
            speed_int = round(speed * 10);
            mean_file = fullfile(dstDir, sprintf('Fukuchi_%d_mean.mat', speed_int));
            
            save(mean_file, 'arr', '-v7');
            
            arr = [var_hip; var_knee; var_ankle];
            
            var_file = fullfile(dstDir, sprintf('Fukuchi_%d_var.mat', speed_int));
            
            save(var_file, 'arr', '-v7');
            
            fprintf('[DEBUG] SAVED: %s, %s\n', mean_file, var_file);
        else
            fprintf('[DEBUG] SKIP: No valid data\n');
        end
    end
    
    fprintf('[DEBUG] FINISHED\n');
end
