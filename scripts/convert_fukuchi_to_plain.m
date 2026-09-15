function convert_fukuchi_to_plain(srcDir, dstDir)
    % convert_fukuchi_to_plain Convert Fukuchi .mat files to plain MAT files.
    %
    % One output file per subject/speed, carrying hip, knee, ankle, Fx, Fy as
    % S.<key>.mean (and .std where the source has it). Read back by
    % gait_loading._load_subject_curve.
    %
    % `dataStruct.variables` is a MATLAB TABLE. Two bugs in the earlier version
    % of this file both came from mishandling that:
    %
    %   1. it indexed `vars{v_idx, 1}`, i.e. the first COLUMN of the row, rather
    %      than the row itself. That reached the name for some variables and not
    %      others, which is why every processed running file came out with hip,
    %      knee, Fx and Fy but NO ANKLE -- while the pooled converter
    %      (generate_fukuchi_consolidated_fixed.m), which uses `vars(v_idx, :)`
    %      and table2struct, extracted the ankle from the same files without
    %      trouble. The ankle was never missing from the Fukuchi data; it was
    %      dropped here. Downstream this silently removed the ankle channel from
    %      every running-speed comparison, since `_valid_channels` treats an
    %      all-NaN reference channel as "not carried by the reference".
    %
    %   2. the name test matched hip_flexion_r and then hip_flexion_l, so the
    %      LEFT leg overwrote the right. Every other source in this project is
    %      right-leg, so the sides now have to agree: variables ending _l are
    %      skipped outright.

    if nargin < 1 || isempty(srcDir)
        srcDir = 'reference_data/fukuchi/2017_Fukuchi_TrackingDataFormat';
    end
    if nargin < 2 || isempty(dstDir)
        dstDir = 'reference_data/fukuchi_processed';
    end

    if ~exist(srcDir, 'dir')
        error('Source directory does not exist: %s', srcDir);
    end
    if ~exist(dstDir, 'dir')
        mkdir(dstDir);
    end

    d = dir(fullfile(srcDir, '*.mat'));
    matFiles = {d.name};
    fprintf('Found %d .mat files under %s\n', numel(matFiles), srcDir);

    nComplete = 0;
    for i = 1:numel(matFiles)
        inFile = fullfile(srcDir, matFiles{i});
        outFile = fullfile(dstDir, matFiles{i});

        try
            data = load(inFile);
            if ~isfield(data, 'dataStruct')
                fprintf('  %s: no dataStruct, skipped\n', matFiles{i});
                continue;
            end
            vars = data.dataStruct.variables;

            S = struct();
            for v_idx = 1:height(vars)
                % Row, not first column -- this is the bug that lost the ankle.
                var_obj = table2struct(vars(v_idx, :));

                if ~isfield(var_obj, 'name')
                    continue;
                end
                name = char(string(var_obj.name));
                if isempty(name)
                    continue;
                end

                % Right leg only; the unsided GRF names pass through.
                if endsWith(name, '_l', 'IgnoreCase', true)
                    continue;
                end

                values = [];
                if isfield(var_obj, 'avg')
                    values = var_obj.avg;
                elseif isfield(var_obj, 'mean')
                    values = var_obj.mean;
                end
                if isempty(values) || ~isnumeric(values)
                    continue;
                end
                if iscolumn(values)
                    values = values';
                end

                nl = lower(name);
                key = '';
                if contains(nl, 'hip') && (contains(nl, 'flexion') || contains(nl, 'angle'))
                    key = 'hip';
                elseif contains(nl, 'knee') && (contains(nl, 'flexion') || contains(nl, 'angle'))
                    key = 'knee';
                elseif contains(nl, 'ankle') && (contains(nl, 'flexion') || ...
                        contains(nl, 'dorsiflexion') || contains(nl, 'angle'))
                    key = 'ankle';
                elseif contains(nl, 'grf') && (contains(nl, 'fore-aft') || contains(nl, '_x'))
                    key = 'Fx';
                elseif contains(nl, 'grf') && (contains(nl, 'vertical') || contains(nl, '_y'))
                    key = 'Fy';
                end
                if isempty(key)
                    continue;
                end

                S.(key).mean = values;
                if isfield(var_obj, 'std')
                    sd = var_obj.std;
                    if iscolumn(sd)
                        sd = sd';
                    end
                    S.(key).std = sd;
                end
            end

            got = fieldnames(S);
            if isempty(got)
                fprintf('  %s: no recognised variables, NOT saved\n', matFiles{i});
                continue;
            end
            % Say what came out, per file: a silently partial conversion is
            % exactly how the ankle went missing for this long.
            wanted = {'hip', 'knee', 'ankle', 'Fx', 'Fy'};
            missing = setdiff(wanted, got);
            if isempty(missing)
                nComplete = nComplete + 1;
            else
                fprintf('  %s: MISSING %s\n', matFiles{i}, strjoin(missing, ', '));
            end
            save(outFile, '-struct', 'S', '-v7');

        catch e
            fprintf('Failed to process %s: %s\n', inFile, e.message);
        end
    end
    fprintf('Conversion complete: %d of %d files carry all five variables.\n', ...
            nComplete, numel(matFiles));
end
