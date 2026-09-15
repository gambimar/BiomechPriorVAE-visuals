function debug_fukuchi_structure()
    % Debug script to understand the exact structure of Fukuchi .mat files
    
    testFile = 'reference_data/fukuchi/2017_Fukuchi_TrackingDataFormat/RBDS001runT25.mat';
    fprintf('Loading: %s\n\n', testFile);
    
    data = load(testFile);
    ds = data.dataStruct(1);
    
    fprintf('dataStruct fields:\n');
    disp(fieldnames(ds));
    
    vars = ds.variables;
    fprintf('\nvariables type: %s\n', class(vars));
    fprintf('variables size (height x width): %d x %d\n', height(vars), width(vars));
    
    if height(vars) > 0
        fprintf('variables columns: %s\n', strjoin(vars.Properties.VariableNames, ', '));
    
    if height(vars) > 0
        fprintf('variables columns: %s\n', strjoin(vars.Properties.VariableNames, ', '));
        
        fprintf('\n--- First 5 variables (detailed) ---\n');
        for i = 1:min(5, height(vars))
            fprintf('\nVariable %d:\n', i);
            
            % Access name
            name_cell = vars.name{i};
            if iscell(name_cell)
                name = name_cell{1};
            else
                name = name_cell;
            end
            fprintf('  name = %s\n', name);
            
            % Access mean
            mean_cell = vars.mean{i};
            if iscell(mean_cell)
                mean_val = mean_cell{1};
            else
                mean_val = mean_cell;
            end
            if isnumeric(mean_val) && numel(mean_val) > 1
                fprintf('  mean = [%d x %d array], sample: [%.4f, %.4f, ...]\n', size(mean_val, 1), size(mean_val, 2), mean_val(1), mean_val(2));
            elseif isnumeric(mean_val)
                fprintf('  mean = %.4f\n', mean_val);
            else
                fprintf('  mean = [%s]\n', class(mean_val));
            end
            
            % Access var
            var_cell = vars.var{i};
            if iscell(var_cell)
                var_val = var_cell{1};
            else
                var_val = var_cell;
            end
            if isnumeric(var_val) && numel(var_val) > 1
                fprintf('  var = [%d x %d array], sample: [%.6f, %.6f, ...]\n', size(var_val, 1), size(var_val, 2), var_val(1), var_val(2));
            elseif isnumeric(var_val)
                fprintf('  var = %.6f\n', var_val);
            else
                fprintf('  var = [%s]\n', class(var_val));
            end
        end
    end
end
