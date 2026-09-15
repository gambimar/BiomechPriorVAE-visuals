% run_baseline_ensemble.m
%
% Imposed-speed PredSim ensembles at every benchmarked speed
% (0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5 m/s), 10 reps each, so the sim side
% of the speed-vs-reference comparison has an actual ensemble (like the
% generative-model benchmarks and the reference subject pool) instead of the
% single deterministic solve baseline_afschrift produced. This is what makes
% a Wasserstein-W1 comparison meaningful rather than a single-point stand-in.
%
% Speed is IMPOSED here (S.bounds.forward_velocity left unset -> PredSim's
% default hard equality at S.misc.forward_velocity), unlike run_free_speed*.m.
%
% Initial guess: PredSim's own 'quasi-random' guess (getGuess_QR_opti.m) is a
% deterministic standing pose -- constant Qs=0 except pelvis_tx (linear ramp)
% and pelvis_ty (constant ride height), constant Qdots. With speed ALSO
% imposed as an equality, 10 reps from that exact guess would hand IPOPT an
% identical problem 10 times over and converge to (near-)bit-identical
% solutions -- zero ensemble spread, useless for W1. So each rep instead
% starts from that same standing pose plus the harmonic-noise perturbation
% run_free_speed_noisy.m already uses on the walking guess (perturb_mot,
% below) -- same style noise, applied to a standing pose instead of a walking
% one, np and pelvis_tx/tz still left alone so distance-traveled and heading
% are not corrupted by noise.
%
% The standing guess itself is synthesized here rather than pulled from
% getGuess_QR_opti.m directly, because that function needs model_info/scaling
% only available deep inside the OCP setup. It is reconstructed from first
% principles instead:
%   - tf(speed): PredSim's own lookup table (all_speeds/all_tf in
%     getGuess_QR_opti.m), reproduced verbatim below.
%   - pelvis_tx: linspace(0, tf*speed, N)   (matches base_forward exactly)
%   - pelvis_ty: constant, taken as the mean pelvis_ty of the shipped walking
%     guess IK_Guess_Full_GC.mot as a stand-in for model_info.IG_pelvis_y
%     (both represent this subject's standing/mean ride height; the exact
%     value does not matter much since every rep is noise-perturbed anyway).
%   - every other coordinate: 0, matching getGuess_QR_opti.m's guess.Qs.
% This is an approximation of the real QR guess (Qdots are derived by
% differentiating the perturbed .mot downstream, same mechanism the walking
% guess already relies on), not a call into PredSim's own function -- flagged
% here in case the two ever need to be reconciled.
%
% Running speeds (2.5/3.5/4.5) reuse the Afschrift et al. bound relaxations
% from run_baseline_afschrift_speeds.m (pelvis_tx Qdot cap, pelvis_tilt
% range, pelvis_ty IG factor) since a mean speed that high needs them
% regardless of where the guess came from. Walking speeds (0.8-1.6) use
% PredSim's default bounds, unmodified -- the same choice run_free_speed_noisy
% made for its walking-range guesses.
%
% Solving from a cold standing guess directly at running speeds (rather than
% warm-starting from a slower converged solve, as run_baseline_afschrift_speeds
% does) is harder for the optimizer and may converge less reliably or take
% longer than the single warm-started baseline_afschrift solves suggest.
% Expect this to run for days, not hours -- restartable by design (see below).

predsim_env();
pathRepo = fileparts(mfilename('fullpath'));
addpath(pathRepo);                              % opensimAD cd's away mid-run
addpath(fullfile(pathRepo,'DefaultSettings'));

resultsRoot = '/home/rzlin/ri94mihu/phd/predsim/results/baseline_ensemble';
igDir       = fullfile(resultsRoot,'ig');
if ~isfolder(igDir); mkdir(igDir); end

SPEEDS  = [0.8 1.0 1.2 1.4 1.6 2.5 3.5 4.5];
SPEED_TAGS = {'v080','v100','v120','v140','v160','v250','v350','v450'};
N_REPS  = 10;

SIGMA_ROT = 0.05;    % rad, per harmonic, on rotational coordinates (~2.9 deg)
SIGMA_TY  = 0.01;    % m,   per harmonic, on pelvis_ty

% Running-speed bound relaxations, from run_baseline_afschrift_speeds.m.
RUNNING_SPEED_MIN      = 2.0;         % speeds >= this get the relaxed bounds
PELVIS_TX_QDOT         = [0, 9.0];    % m/s   (default [0.0278 2.7304])
PELVIS_TILT_QS         = [-45, 20];   % deg   (default [-20.0 3.9])
PELVIS_TY_FACTOR_UPPER = 1.5;         % x IG  (default 1.2)

baseIG = fullfile(pathRepo,'OCP','IK_Guess_Full_GC.mot');

fid = fopen(fullfile(resultsRoot,'run_notes.txt'),'w');
fprintf(fid,'Imposed-speed PredSim ensembles, %d reps per speed, at: %s m/s\n\n', ...
    N_REPS, mat2str(SPEEDS));
fprintf(fid,'Speed is imposed (default hard equality at S.misc.forward_velocity),\n');
fprintf(fid,'unlike run_free_speed*.m where it is a free variable.\n\n');
fprintf(fid,'Initial guess: synthesized standing pose (Qs=0 except pelvis_tx ramp\n');
fprintf(fid,'and constant pelvis_ty), approximating PredSim''s own quasi-random\n');
fprintf(fid,'guess (getGuess_QR_opti.m), then perturbed per rep with rng(seed):\n');
fprintf(fid,'  q_i(t) <- q_i(t) + a_i + b_i*sin(2*pi*t/T) + c_i*cos(2*pi*t/T)\n');
fprintf(fid,'  a,b,c ~ N(0, sigma^2), sigma = %g rad (rotations), %g m (pelvis_ty)\n', ...
    SIGMA_ROT, SIGMA_TY);
fprintf(fid,'  pelvis_tx and pelvis_tz left untouched.\n\n');
fprintf(fid,'Running speeds (>= %g m/s) use the Afschrift et al. bound relaxations:\n', ...
    RUNNING_SPEED_MIN);
fprintf(fid,'  S.bounds.Qdots pelvis_tx   = [%g %g] m/s (default [0.0278 2.7304])\n', PELVIS_TX_QDOT);
fprintf(fid,'  S.bounds.Qs    pelvis_tilt = [%g %g] deg (default [-20.0 3.9])\n', PELVIS_TILT_QS);
fprintf(fid,'  S.bounds.factor_IG_pelvis_ty.upper = %g (default 1.2)\n', PELVIS_TY_FACTOR_UPPER);
fprintf(fid,'Walking speeds use PredSim''s default bounds, unmodified.\n\n');
fprintf(fid,'Otherwise the Afschrift et al. settings used throughout this project:\n');
fprintf(fid,'Falisse_et_al_2022 model, bounds.a.lower 0.01, N_meshes 50, N_threads 2.\n');
fclose(fid);

timingFile = fullfile(resultsRoot,'timings.csv');
if ~isfile(timingFile)
    fid = fopen(timingFile,'w');
    fprintf(fid,['speed_mps,rep,seed,wallclock_s,wallclock_min,success,return_status,' ...
                 'iterations,objective,vel_achieved_mps,savename\n']);
    fclose(fid);
end

for si = 1:numel(SPEEDS)
    speed = SPEEDS(si);
    speedTag = SPEED_TAGS{si};
    isRunning = speed >= RUNNING_SPEED_MIN;

    standingMot = fullfile(igDir, sprintf('%s_standing.mot', speedTag));
    make_standing_mot(baseIG, standingMot, speed);

    for rep = 1:N_REPS
        tag = sprintf('%s_rep%02d', speedTag, rep);

        % skip combos already logged, so the script is restartable after a
        % crash or a lost license seat without redoing hours of solving
        logged = strsplit(fileread(timingFile), newline);
        logged = logged(~cellfun(@isempty, strtrim(logged)));
        done = any(startsWith(logged, sprintf('%.2f,%d,', speed, rep)));
        if done
            fprintf('=== %s already in timings.csv, skipping ===\n', tag);
            continue
        end

        seed = si*100 + rep;   % unique per (speed, rep), reproducible on its own
        rng(seed, 'twister');
        igFile = fullfile(igDir, sprintf('%s.mot', tag));
        perturb_mot(standingMot, igFile, SIGMA_ROT, SIGMA_TY);

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

fprintf('\n=== BASELINE ENSEMBLE RUNS COMPLETE. Timings: %s ===\n', timingFile);


function make_standing_mot(templateFile, outFile, speed)
% Synthesize a standing-pose .mot guess for `speed`: pelvis_tx ramps from 0 to
% tf*speed, pelvis_ty is held at the template's mean ride height, every other
% coordinate is 0. `templateFile` supplies the header/column layout only --
% none of its motion data is kept. tf(speed) reproduces the lookup table in
% PredSim/OCP/getGuess_QR_opti.m verbatim.

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
t_template = data(:,1);
T = t_template(end) - t_template(1);
if T <= 0; T = 1; end

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

pelvis_ty_idx = find(strcmp(labels,'pelvis_ty'));
pelvis_tx_idx = find(strcmp(labels,'pelvis_tx'));
assert(~isempty(pelvis_ty_idx) && ~isempty(pelvis_tx_idx), 'pelvis_tx/ty missing in %s', templateFile);
standing_ty = mean(data(:,pelvis_ty_idx));

t = linspace(0, T, N)';   % rescale template's own time axis; content is discarded below
data = zeros(N, numel(labels));
data(:,1) = t;
data(:,pelvis_tx_idx) = linspace(0, tf*speed, N);
data(:,pelvis_ty_idx) = standing_ty;

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
% run_free_speed_noisy.m (duplicated rather than shared, matching that
% script's own choice to keep each run script self-contained).

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
