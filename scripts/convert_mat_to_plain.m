function convert_mat_to_plain(srcDir, dstDir)
% convert_mat_to_plain Recursively convert .mat files to plain, non-class MAT files
%
% Usage (from terminal):
% matlab -batch "convert_mat_to_plain('/path/to/src','/path/to/dst')"
% or from MATLAB:
% convert_mat_to_plain('/path/to/src','/path/to/dst')
%
% The script loads each .mat file under srcDir, converts MATLAB class
% objects to plain MATLAB structs where possible, converts tables to
% structs (columns become fields), converts cells and nested structs
% recursively, and saves the result as a MATLAB v7 file under dstDir
% preserving the directory tree. v7 files are readable by Python
% libraries such as mat-io and scipy (avoid v7.3/HDF5).



if nargin < 1 || isempty(srcDir)
    srcDir = 'results_sim/simulations';
end
if nargin < 2 || isempty(dstDir)
    dstDir = 'results_sim/converted';
end
srcDir = char(srcDir);
dstDir = char(dstDir);

if ~exist(srcDir, 'dir')
    error('Source directory does not exist: %s', srcDir);
end
if ~exist(dstDir, 'dir')
    mkdir(dstDir);
end

% Some saved results were solved with contact-stiffness/damping variant
% classes (Gait3d_smoothsphere_softdamp2/4, _stiffer2/4) -- their real
% classdefs (fetched from the workstation) now live alongside
% Gait3d_smoothsphere itself under BioMAC-Sim-Toolbox/src/model/gait3d/, and
% their MEX was rebuilt for this machine (see convert_mac_mex.m). No local
% stub/addpath needed here anymore.

matFiles = list_mat_files(srcDir);
fprintf('Found %d .mat files under %s\n', numel(matFiles), srcDir);
count = 0;
for i = 1:numel(matFiles)
    inFile = matFiles{i};
    rel = strrep(inFile, [srcDir filesep], [dstDir filesep]);
    outFile = fullfile(rel);
    outDir = fileparts(outFile);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    if exist(outFile, 'file')
        fprintf('Output file already exists, skipping: %s\n', outFile);
        continue;
    end
    try
        data = load(inFile);
    catch e
        fprintf('Failed to load %s: %s\n', inFile, e.message);
        continue;
    end

    S = struct();
    vars = fieldnames(data);
    if isempty(vars)
        fprintf('No variables in %s, skipping.\n', inFile);
        continue;
    end

    S.converged = data.result.converged;
    S.info = data.result.info;
    S.X = data.result.X;
    S.objectives = data.result.problem.objectiveTerms;
    S.constraints = data.result.problem.constraintTerms;
    if size(S.X,1) <= 20000
        inFile
    end

    if size(S.X,1) > 22500
        S.dur = data.result.X(22951);
    elseif size(S.X,1) > 22000
        S.dur = data.result.X(22401);
    elseif size(S.X,1) > 21000
        S.dur = data.result.X(21201);
    end
    % getMetabolicCost dispatches to a compiled MEX matching the model's own
    % internal name string (e.g. gait3d_pelvis213_smoothsphere_softdamp2),
    % not its MATLAB class -- a contact-variant model whose MEX was never
    % built/copied onto this machine fails here with no workaround short of
    % obtaining that compiled MEX (the classdef stub trick above only fixes
    % *loading* the object, not evaluating its compiled dynamics). Caught
    % per-file so one missing MEX doesn't abort the whole batch; the file is
    % skipped (not partially saved) so a downstream consumer never sees a
    % converted file silently missing its metabolicCost* columns.
    try
        S.metabolicCost = data.result.problem.getMetabolicCost(data.result.X);
        S.metabolicCostBhargava = data.result.problem.getMetabolicCost(data.result.X,'bhargava');
        S.metabolicCostUmberger = data.result.problem.getMetabolicCost(data.result.X,'umberger');
        S.metabolicCostHoudijk = data.result.problem.getMetabolicCost(data.result.X,'houdijk');
        S.metabolicCostMargaria = data.result.problem.getMetabolicCost(data.result.X,'margaria');
        S.metabolicCostMinetti = data.result.problem.getMetabolicCost(data.result.X,'minetti');
        S.metabolicCostLichtwark = data.result.problem.getMetabolicCost(data.result.X,'lichtwark');
        S.metabolicCostKim = data.result.problem.getMetabolicCost(data.result.X,'kim');
    catch e
        fprintf('Failed to compute metabolic cost for %s (likely a missing model-variant MEX): %s\n', inFile, e.message);
        continue;
    end



    save(outFile, '-struct', 'S', '-v7');
    fprintf('Saved converted file: %s\n', outFile);
    count = count + 1;
end


fprintf('Conversion complete — %d files written to %s\n', count, dstDir);
end


function files = list_mat_files(root)
    % Use recursive dir if available (R2016b+), otherwise walk manually
    try
        d = dir(fullfile(root, '**', '*.mat'));
        files = fullfile({d.folder}, {d.name});
    catch
        files = {};
        stack = {root};
        while ~isempty(stack)
            cur = stack{end}; stack(end) = [];
            listing = dir(cur);
            for i = 1:numel(listing)
                item = listing(i);
                if item.isdir && ~startsWith(item.name, '.')
                    stack{end+1} = fullfile(cur, item.name);
                elseif ~item.isdir && endsWith(item.name, '.mat')
                    files{end+1} = fullfile(cur, item.name); %#ok<AGROW>
                end
            end
        end
    end
end

function out = convert_value(x)
    % Recursively convert MATLAB values to plain MATLAB types
    persistent MAX_DEPTH
    if isempty(MAX_DEPTH)
        MAX_DEPTH = 20;
    end
    out = convert_value_impl(x, 0, MAX_DEPTH);
end

function out = convert_value_impl(x, depth, maxDepth)
    if depth > maxDepth
        out = [];
        return;
    end

    % Tables: convert to scalar struct where each variable becomes a field
    if istable(x)
        try
            out = table2struct(x, 'ToScalar', true);
            % ensure fields are converted recursively
            fn = fieldnames(out);
            for i = 1:numel(fn)
                out.(fn{i}) = convert_value_impl(out.(fn{i}), depth+1, maxDepth);
            end
            return;
        catch
            % fallback: convert columns manually
            vars = x.Properties.VariableNames;
            out = struct();
            for i = 1:numel(vars)
                out.(vars{i}) = convert_value_impl(x.(vars{i}), depth+1, maxDepth);
            end
            return;
        end
    end

    % MATLAB objects / class instances
    if isobject(x)
        try
            s = struct(x); % often converts objects to struct of properties
        catch
            % try reading public properties
            try
                pnames = properties(x);
                s = struct();
                for i = 1:numel(pnames)
                    try
                        s.(pnames{i}) = x.(pnames{i});
                    catch
                        s.(pnames{i}) = [];
                    end
                end
            catch
                error('Cannot convert object of class %s', class(x));
            end
        end
        % convert fields recursively
        fn = fieldnames(s);
        for i = 1:numel(fn)
            s.(fn{i}) = convert_value_impl(s.(fn{i}), depth+1, maxDepth);
        end
        out = s;
        return;
    end

    % Structs: recurse into fields and struct arrays
    if isstruct(x)
        sz = size(x);
        if numel(x) > 1
            % struct array: convert each element
            out = repmat(struct(), sz);
            for idx = 1:numel(x)
                tmp = struct();
                fn = fieldnames(x(idx));
                for i = 1:numel(fn)
                    tmp.(fn{i}) = convert_value_impl(x(idx).(fn{i}), depth+1, maxDepth);
                end
                out(idx) = tmp;
            end
            return;
        else
            out = struct();
            fn = fieldnames(x);
            for i = 1:numel(fn)
                out.(fn{i}) = convert_value_impl(x.(fn{i}), depth+1, maxDepth);
            end
            return;
        end
    end

    % Cells: convert each element
    if iscell(x)
        out = cell(size(x));
        for i = 1:numel(x)
            out{i} = convert_value_impl(x{i}, depth+1, maxDepth);
        end
        return;
    end

    % Function handles: convert to string
    if isa(x, 'function_handle')
        out = func2str(x);
        return;
    end

    % Numeric, char, logical, etc. — return as-is
    out = x;
end

function tf = is_value_empty(v)
    % Return true if value is empty or contains only empty fields recursively
    if isempty(v)
        tf = true; return;
    end
    if isstruct(v)
        fn = fieldnames(v);
        if isempty(fn)
            tf = true; return;
        end
        for i = 1:numel(v)
            for j = 1:numel(fn)
                if ~is_value_empty(v(i).(fn{j}))
                    tf = false; return;
                end
            end
        end
        tf = true; return;
    end
    if iscell(v)
        for i = 1:numel(v)
            if ~is_value_empty(v{i})
                tf = false; return;
            end
        end
        tf = true; return;
    end
    % For other types (numeric, char, logical, objects) consider non-empty
    tf = false;
end
