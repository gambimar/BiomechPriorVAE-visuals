function run_baseline_ensemble_subset(SPEEDS)
% run_baseline_ensemble_subset(SPEEDS) -- run_baseline_ensemble.m restricted
% to the given SPEEDS subset, so several instances can run concurrently
% (different license seats) against the SAME resultsRoot/timings.csv instead
% of one process working through all 8 speeds sequentially.
%
% Launched via: matlab -batch "run_baseline_ensemble_subset([1.2 1.4])"
%
% Shares timings.csv with every other instance -- the restart-skip logic
% below means whichever instance reaches a given (speed,rep) first "wins"
% and the others skip it once they check, so running overlapping speed sets
% is wasteful but not unsafe. Seeds are derived from the FULL 8-speed
% benchmark list (not this subset's own index), so a given (speed,rep)
% always gets the same seed regardless of which instance solves it.
%
% resultsRoot is 'baseline_ensemble_v2', a fresh results tree separate from
% the original 'baseline_ensemble' -- v1 used a synthesized, all-zero-except-
% pelvis standing pose as the base guess, which turned out to let the
% optimizer escape into "period-doubled" local optima at a majority of
% walking speeds (dur landing exactly on S.bounds.t_final.upper = 2s instead
% of a genuine single-stride cycle -- confirmed against the shipped
% Falisse2022 benchmark's own cycle durations, which are 2-3x shorter).
% v2's guess (make_realistic_ig_mot, below) uses REAL kinematic data instead
% of a static pose, which is expected to make the true single-stride
% solution far more reachable. Kept separate from v1's tree rather than
% overwriting it so the two are directly comparable and v1's already-good
% solves (rare, but present) aren't silently discarded.
%
% See run_baseline_ensemble.m for the fuller rationale (running-speed bound
% relaxations etc.) -- this is that script with the SPEEDS/SPEED_TAGS loop
% parameterized and the per-instance preamble (run_notes.txt) dropped since
% it isn't the source of truth when several instances share one resultsRoot.

predsim_env();
pathRepo = fileparts(mfilename('fullpath'));
addpath(pathRepo);                              % opensimAD cd's away mid-run
addpath(fullfile(pathRepo,'DefaultSettings'));

resultsRoot = '/home/rzlin/ri94mihu/phd/predsim/results/baseline_ensemble_v2';
igDir       = fullfile(resultsRoot,'ig');
if ~isfolder(igDir); mkdir(igDir); end

fukuchiDir = fullfile(pathRepo,'reference_data_ig');   % Fukuchi_{25,35,45}_mean.mat

FULL_SPEEDS = [0.8 1.0 1.2 1.4 1.6 2.5 3.5 4.5];  % seed derivation only
N_REPS  = 10;

SIGMA_ROT = 0.05;    % rad, per harmonic, on rotational coordinates (~2.9 deg)
SIGMA_TY  = 0.01;    % m,   per harmonic, on pelvis_ty

% Running-speed bound relaxations, from run_baseline_afschrift_speeds.m.
RUNNING_SPEED_MIN      = 2.0;         % speeds >= this get the relaxed bounds
PELVIS_TX_QDOT         = [0, 9.0];    % m/s   (default [0.0278 2.7304])
PELVIS_TILT_QS         = [-45, 20];   % deg   (default [-20.0 3.9])
PELVIS_TY_FACTOR_UPPER = 1.5;         % x IG  (default 1.2)

baseIG = fullfile(pathRepo,'OCP','IK_Guess_Full_GC.mot');

timingFile = fullfile(resultsRoot,'timings.csv');
if ~isfile(timingFile)
    fid = fopen(timingFile,'w');
    fprintf(fid,['speed_mps,rep,seed,wallclock_s,wallclock_min,success,return_status,' ...
                 'iterations,objective,vel_achieved_mps,savename\n']);
    fclose(fid);
end

for si = 1:numel(SPEEDS)
    speed = SPEEDS(si);
    speedTag = sprintf('v%03d', round(speed*100));
    isRunning = speed >= RUNNING_SPEED_MIN;

    baseMot = fullfile(igDir, sprintf('%s_base.mot', speedTag));
    make_realistic_ig_mot(baseIG, fukuchiDir, baseMot, speed);

    for rep = 1:N_REPS
        tag = sprintf('%s_rep%02d', speedTag, rep);

        % skip combos already logged, so the script is restartable after a
        % crash or a lost license seat without redoing hours of solving --
        % also what lets multiple concurrent instances share one timings.csv
        % without duplicating a completed (speed,rep).
        logged = strsplit(fileread(timingFile), newline);
        logged = logged(~cellfun(@isempty, strtrim(logged)));
        done = any(startsWith(logged, sprintf('%.2f,%d,', speed, rep)));
        if done
            fprintf('=== %s already in timings.csv, skipping ===\n', tag);
            continue
        end

        si_full = find(round(FULL_SPEEDS*100) == round(speed*100), 1);
        seed = si_full*100 + rep;   % unique per (speed, rep), matches run_baseline_ensemble.m
        rng(seed, 'twister');
        igFile = fullfile(igDir, sprintf('%s.mot', tag));
        perturb_mot(baseMot, igFile, SIGMA_ROT, SIGMA_TY);

        cd(pathRepo);   % opensimAD leaves cwd inside its build tree
        [S] = initializeSettings('Falisse_et_al_2022');
        S.subject.name          = 'Falisse_et_al_2022';
        S.misc.forward_velocity = speed;              % imposed (hard equality)
        S.misc.save_folder      = fullfile(resultsRoot, tag);

        S.bounds.a.lower          = 0.01;
        S.solver.N_meshes         = 50;
        S.solver.N_threads        = 2;
        S.solver.run_as_batch_job = false;

        if isRunning
            S.bounds.Qdots = {'pelvis_tx',   PELVIS_TX_QDOT(1), PELVIS_TX_QDOT(2)};
            S.bounds.Qs    = {'pelvis_tilt', PELVIS_TILT_QS(1), PELVIS_TILT_QS(2)};
            S.bounds.factor_IG_pelvis_ty.upper = PELVIS_TY_FACTOR_UPPER;
        end

        S.solver.IG_selection = igFile;
        S.solver.IG_selection_gaitCyclePercent = 100;

        osim_path = fullfile(pathRepo,'Subjects',S.subject.name,[S.subject.name '.osim']);

        fprintf('\n=== %s (speed %.2f m/s, seed %d) START %s ===\n', ...
            tag, speed, seed, datestr(now));
        ok = true; savename = '';
        t0 = tic;
        try
            savename = runPredSim(S, osim_path);
        catch err
            ok = false;
            fprintf('!!! RUN FAILED for %s: %s\n', tag, err.message);
        end
        el = toc(t0);
        fprintf('=== %s DONE in %.1f s (%.2f min) ===\n', tag, el, el/60);

        success = NaN; retstat = 'n/a'; iters = NaN; objv = NaN; velach = NaN;
        if ok
            resFile = fullfile(S.misc.save_folder,[savename '.mat']);
            if isfile(resFile)
                D = load(resFile);
                if isfield(D,'stats')
                    if isfield(D.stats,'success');       success = double(D.stats.success); end
                    if isfield(D.stats,'return_status'); retstat = D.stats.return_status;  end
                    if isfield(D.stats,'iter_count');    iters   = D.stats.iter_count;     end
                    if isfield(D.stats,'iterations') && isfield(D.stats.iterations,'obj')
                        objv = D.stats.iterations.obj(end);
                    end
                end
                if isfield(D,'R') && isfield(D.R,'spatiotemp') ...
                        && isfield(D.R.spatiotemp,'dist_trav') && isfield(D.R,'time')
                    velach = D.R.spatiotemp.dist_trav / D.R.time.mesh_GC(end);
                end
            end
        end

        fid = fopen(timingFile,'a');
        fprintf(fid,'%.2f,%d,%d,%.1f,%.3f,%g,%s,%g,%g,%g,%s\n', ...
            speed, rep, seed, el, el/60, success, retstat, iters, objv, velach, savename);
        fclose(fid);

        fprintf('--> %s achieved speed: %.4f m/s (imposed %.2f)\n', tag, velach, speed);
    end
end

fprintf('\n=== SUBSET RUN COMPLETE (%s). Timings: %s ===\n', mat2str(SPEEDS), timingFile);
end


function make_realistic_ig_mot(templateFile, fukuchiDir, outFile, speed)
% Build a periodic-gait initial guess for `speed` from REAL kinematic data,
% replacing v1's synthesized all-zero-except-pelvis standing pose:
%
%   walking speeds (< RUNNING_SPEED_MIN): the shipped mocap walking cycle
%     (`templateFile`, a real ~1.3 m/s stride) is used as-is, EXCEPT its time
%     axis is rescaled to tf(speed) (PredSim's own lookup, verbatim from
%     getGuess_QR_opti.m) and pelvis_tx is rescaled to travel tf*speed --
%     i.e. the waveform SHAPE (every joint vs. %gait-cycle) is kept exactly,
%     only the time/distance axes are stretched to the target speed. Every
%     other column (pelvis_ty's real vertical bob, arm swing, lumbar, etc.)
%     is carried through unchanged in shape, which alone should already be a
%     far better starting point than a static pose for the optimizer to
%     linearize around.
%
%   running speeds (>= RUNNING_SPEED_MIN): same time/distance rescale of the
%     walking template as the base (so the rest of the body still has a
%     periodic, non-zero guess), but hip_flexion/knee_angle/ankle_angle
%     (both legs) are overridden with the Fukuchi running reference at the
%     nearest of {2.5, 3.5, 4.5} m/s -- the only DOFs real running
%     kinematics are available for in this repo (reference_data/Fukuchi_*_
%     mean.mat, hip/knee/ankle only, one representative leg). Left leg is
%     the right leg's curve rolled by half a cycle (bilateral-symmetry
%     assumption). Rows are radians already (same convention as this file's
%     own coordinates -- confirmed knee is negative-signed in both,
%     matching this project's flexion convention, so no sign flip needed).

RUNNING_SPEED_MIN = 2.0;

fidIn = fopen(templateFile,'r');
raw = textscan(fidIn,'%s','Delimiter','\n','Whitespace','');
fclose(fidIn);
lines = raw{1};

iEnd = find(strcmp(strtrim(lines),'endheader'),1);
assert(~isempty(iEnd), 'no endheader in %s', templateFile);

labels = strsplit(strtrim(lines{iEnd+1}), sprintf('\t'));
labels = cellfun(@strtrim, labels, 'UniformOutput', false);
labels = labels(~cellfun(@isempty,labels));

data = cell2mat(cellfun(@(L) sscanf(L,'%f')', lines(iEnd+2:end), 'UniformOutput', false));
assert(size(data,2) == numel(labels), 'label/column mismatch in %s', templateFile);

N = size(data,1);

% tf(speed) lookup, verbatim from getGuess_QR_opti.m
all_speeds = 0.73:0.1:5;
all_tf = 0.70:-((0.70-0.35)/(length(all_speeds)-1)):0.35;
idx_speed = find(all_speeds==speed);
if isempty(idx_speed)
    idx_speed = find(all_speeds > speed,1,'first');
end
if isempty(idx_speed)
    tf = -0.1750*speed + 0.8277;   % extrapolate outside 0.73:5
else
    tf = all_tf(idx_speed);
end
if tf < 0.15; tf = 0.15; end

pelvis_tx_idx = find(strcmp(labels,'pelvis_tx'));
assert(~isempty(pelvis_tx_idx), 'pelvis_tx missing in %s', templateFile);

data(:,1) = linspace(0, tf, N)';                     % time axis rescaled to tf(speed)
data(:,pelvis_tx_idx) = linspace(0, tf*speed, N)';   % distance rescaled to hit the imposed speed

if speed >= RUNNING_SPEED_MIN
    speed_tags  = [2.5 3.5 4.5];
    speed_files = {'Fukuchi_25_mean.mat','Fukuchi_35_mean.mat','Fukuchi_45_mean.mat'};
    [~, fi] = min(abs(speed_tags - speed));
    fdata = load(fullfile(fukuchiDir, speed_files{fi}));
    arr = fdata.arr;   % 3 x N radians, rows unidentified -- same heuristic as
                        % evaluation/data_utils.py's get_reference_data:
                        % knee has the largest range, then hip > ankle.
    assert(size(arr,2) == N, ...
        'Fukuchi curve length %d does not match template N=%d', size(arr,2), N);
    ranges = max(abs(arr), [], 2);
    [~, knee_row] = max(ranges);
    other = setdiff(1:3, knee_row);
    if ranges(other(1)) > ranges(other(2))
        hip_row = other(1); ankle_row = other(2);
    else
        hip_row = other(2); ankle_row = other(1);
    end

    half = round(N/2);
    hip_r   = arr(hip_row,:)';    hip_l   = circshift(hip_r,   half);
    knee_r  = arr(knee_row,:)';   knee_l  = circshift(knee_r,  half);
    ankle_r = arr(ankle_row,:)';  ankle_l = circshift(ankle_r, half);

    data(:, strcmp(labels,'hip_flexion_r'))  = hip_r;
    data(:, strcmp(labels,'hip_flexion_l'))  = hip_l;
    data(:, strcmp(labels,'knee_angle_r'))   = knee_r;
    data(:, strcmp(labels,'knee_angle_l'))   = knee_l;
    data(:, strcmp(labels,'ankle_angle_r'))  = ankle_r;
    data(:, strcmp(labels,'ankle_angle_l'))  = ankle_l;
end

fidOut = fopen(outFile,'w');
for i = 1:iEnd
    fprintf(fidOut,'%s\n',lines{i});
end
fprintf(fidOut,'%s\t',labels{:});
fprintf(fidOut,'\n');
for r = 1:size(data,1)
    fprintf(fidOut,'%20.8f\t',data(r,:));
    fprintf(fidOut,'\n');
end
fclose(fidOut);
end


function perturb_mot(inFile, outFile, sigmaRot, sigmaTy)
% Add periodic noise to every coordinate column of an OpenSim .mot guess and
% write it back out in the same format. Identical to the helper in
% run_free_speed_noisy.m / run_baseline_ensemble.m (duplicated rather than
% shared, matching those scripts' own choice to keep each run self-contained).

fidIn = fopen(inFile,'r');
raw = textscan(fidIn,'%s','Delimiter','\n','Whitespace','');
fclose(fidIn);
lines = raw{1};

iEnd = find(strcmp(strtrim(lines),'endheader'),1);
assert(~isempty(iEnd), 'no endheader in %s', inFile);

labels = strsplit(strtrim(lines{iEnd+1}), sprintf('\t'));
labels = cellfun(@strtrim, labels, 'UniformOutput', false);
labels = labels(~cellfun(@isempty,labels));

data = cell2mat(cellfun(@(L) sscanf(L,'%f')', lines(iEnd+2:end), 'UniformOutput', false));
assert(size(data,2) == numel(labels), 'label/column mismatch in %s', inFile);

t = data(:,1);
T = t(end) - t(1);
if T <= 0; T = 1; end
phase = 2*pi*(t - t(1))/T;

for c = 2:size(data,2)
    name = labels{c};
    switch name
        case {'pelvis_tx','pelvis_tz'}
            continue                       % heading and travelled distance
        case 'pelvis_ty'
            s = sigmaTy;
        otherwise
            s = sigmaRot;                  % all rotations, in rad
    end
    a = s*randn(); b = s*randn(); c3 = s*randn();
    data(:,c) = data(:,c) + a + b*sin(phase) + c3*cos(phase);
end

fidOut = fopen(outFile,'w');
for i = 1:iEnd
    fprintf(fidOut,'%s\n',lines{i});
end
fprintf(fidOut,'%s\t',labels{:});
fprintf(fidOut,'\n');
for r = 1:size(data,1)
    fprintf(fidOut,'%20.8f\t',data(r,:));
    fprintf(fidOut,'\n');
end
fclose(fidOut);
end
