% run_free_speed_noisy.m
%
% Free-speed PredSim, 10 repetitions, each started from a DIFFERENT randomly
% perturbed initial guess.
%
% Why: run_free_speed.m probes the free-speed optimum from three clean initial
% guesses that differ only in the seeded forward velocity. That varies one
% scalar of the guess. If the emergent speed is a property of the model rather
% than of the starting point, it must also survive perturbation of the whole
% guessed TRAJECTORY. Ten repetitions with noise on every coordinate give a
% distribution of emergent speeds instead of three points, so "the runs agree"
% becomes a statement with a spread attached.
%
% Speed is free in [0.5, 2.0] m/s (wider at the bottom than run_free_speed.m's
% 0.6, so the lower bound cannot be mistaken for the optimum).
%
% Noise model: the shipped walking IG (IK_Guess_Full_GC.mot) is perturbed per
% coordinate by a constant offset plus one sine and one cosine harmonic of the
% gait cycle. Harmonics are used rather than white noise because the guess is
% differentiated to seed Qdots -- white noise would produce absurd velocities
% and the perturbation would be a numerical artefact rather than a different
% starting posture. Being harmonics of the cycle, they are also exactly
% periodic, so the guess still closes on itself as a gait cycle.
%
% Translations pelvis_tx/tz are NOT perturbed: tx carries the travelled
% distance that sets the seeded speed (which is randomised separately and
% explicitly), and tz would tilt the heading. pelvis_ty (ride height) is
% perturbed on a smaller scale, in metres.

predsim_env();
pathRepo = fileparts(mfilename('fullpath'));
addpath(pathRepo);                              % opensimAD cd's away mid-run
addpath(fullfile(pathRepo,'DefaultSettings'));

resultsRoot = '/home/rzlin/ri94mihu/phd/predsim/results/free_speed_noisy';
igDir       = fullfile(resultsRoot,'ig');
if ~isfolder(igDir); mkdir(igDir); end

N_REPS  = 10;
V_LOWER = 0.5;
V_UPPER = 2.0;

SIGMA_ROT = 0.05;    % rad, per harmonic, on rotational coordinates (~2.9 deg)
SIGMA_TY  = 0.01;    % m,   per harmonic, on pelvis_ty

% See run_free_speed.m for why this cap is raised: the default 2.7304 m/s cap
% on INSTANTANEOUS pelvis_tx velocity would bind near the 2.0 m/s upper bound
% and depress the emergent speed as an artefact of the bound.
PELVIS_TX_QDOT = [0, 4.0];

baseIG = fullfile(pathRepo,'OCP','IK_Guess_Full_GC.mot');

% ---- reproducibility -------------------------------------------------------
% Each rep seeds the generator with its own index, so rep k is reproducible on
% its own and re-running a single failed rep reproduces exactly its guess.

fid = fopen(fullfile(resultsRoot,'run_notes.txt'),'w');
fprintf(fid,'Free-speed PredSim with a NOISY initial guess, %d repetitions.\n\n',N_REPS);
fprintf(fid,'Average velocity constrained to [%g, %g] m/s (no equality).\n',V_LOWER,V_UPPER);
fprintf(fid,'Cost is per unit distance (Jall_sc), so the optimum is the\n');
fprintf(fid,'cost-of-transport-optimal speed.\n\n');
fprintf(fid,'Guess perturbation, rep k with rng(k):\n');
fprintf(fid,'  q_i(t) <- q_i(t) + a_i + b_i*sin(2*pi*t/T) + c_i*cos(2*pi*t/T)\n');
fprintf(fid,'  a,b,c ~ N(0, sigma^2), sigma = %g rad (rotations), %g m (pelvis_ty)\n', ...
    SIGMA_ROT, SIGMA_TY);
fprintf(fid,'  pelvis_tx and pelvis_tz left untouched.\n');
fprintf(fid,'  seeded S.misc.forward_velocity ~ U(%g, %g) m/s (guess only)\n\n',V_LOWER,V_UPPER);
fprintf(fid,'Bound override: S.bounds.Qdots pelvis_tx = [%g %g] m/s (default [0.0278 2.7304])\n', ...
    PELVIS_TX_QDOT);
fprintf(fid,'Otherwise the Afschrift et al. settings used for the speed baselines:\n');
fprintf(fid,'Falisse_et_al_2022 model, bounds.a.lower 0.01, N_meshes 50, N_threads 2.\n');
fclose(fid);

timingFile = fullfile(resultsRoot,'timings.csv');
if ~isfile(timingFile)
    fid = fopen(timingFile,'w');
    fprintf(fid,['rep,seed,v_guess,wallclock_s,wallclock_min,success,return_status,' ...
                 'iterations,objective,vel_emergent_mps,savename\n']);
    fclose(fid);
end

% ---- generate every perturbed guess before solving anything ----
% Up front rather than inside the solve loop so a format problem in
% perturb_mot fails in seconds instead of after the first multi-hour solve,
% and so rep k's guess does not depend on how many reps ran before it.
igFiles  = cell(N_REPS,1);
v_guesses = zeros(N_REPS,1);
for rep = 1:N_REPS
    rng(rep,'twister');
    igFiles{rep}   = fullfile(igDir,sprintf('rep%02d.mot',rep));
    v_guesses(rep) = V_LOWER + (V_UPPER-V_LOWER)*rand();
    perturb_mot(baseIG, igFiles{rep}, SIGMA_ROT, SIGMA_TY);
end
fprintf('Generated %d perturbed initial guesses in %s\n', N_REPS, igDir);
fprintf('Seeded velocities: %s\n', mat2str(round(v_guesses',3)));

for rep = 1:N_REPS
    tag = sprintf('rep%02d',rep);

    % skip reps already logged, so the script is restartable after a crash or
    % a lost license seat without redoing hours of solving
    logged = strsplit(fileread(timingFile), newline);
    logged = logged(~cellfun(@isempty, strtrim(logged)));
    done = any(startsWith(logged, sprintf('%d,',rep)));
    if done
        fprintf('=== %s already in timings.csv, skipping ===\n',tag);
        continue
    end

    igFile  = igFiles{rep};
    v_guess = v_guesses(rep);

    cd(pathRepo);   % opensimAD leaves cwd inside its build tree
    [S] = initializeSettings('Falisse_et_al_2022');
    S.subject.name          = 'Falisse_et_al_2022';
    S.misc.forward_velocity = v_guess;    % seeds tf/Qdots only, not a constraint
    S.misc.save_folder      = fullfile(resultsRoot, tag);

    S.bounds.forward_velocity.lower = V_LOWER;
    S.bounds.forward_velocity.upper = V_UPPER;

    S.bounds.a.lower          = 0.01;
    S.solver.N_meshes         = 50;
    S.solver.N_threads        = 2;
    S.solver.run_as_batch_job = false;

    S.bounds.Qdots = {'pelvis_tx', PELVIS_TX_QDOT(1), PELVIS_TX_QDOT(2)};

    S.solver.IG_selection = igFile;
    S.solver.IG_selection_gaitCyclePercent = 100;

    osim_path = fullfile(pathRepo,'Subjects',S.subject.name,[S.subject.name '.osim']);

    fprintf('\n=== %s (seed %d, v_guess %.3f m/s) START %s ===\n', ...
        tag, rep, v_guess, datestr(now));
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
                    && isfield(D.R.spatiotemp,'vel_aver')
                velach = D.R.spatiotemp.vel_aver;
            end
        end
    end

    fid = fopen(timingFile,'a');
    fprintf(fid,'%d,%d,%.4f,%.1f,%.3f,%g,%s,%g,%g,%g,%s\n', ...
        rep, rep, v_guess, el, el/60, success, retstat, iters, objv, velach, savename);
    fclose(fid);

    fprintf('--> %s emergent speed: %.4f m/s (guess was %.3f)\n', tag, velach, v_guess);
end

fprintf('\n=== NOISY FREE-SPEED RUNS COMPLETE. Timings: %s ===\n', timingFile);


function perturb_mot(inFile, outFile, sigmaRot, sigmaTy)
% Add periodic noise to every coordinate column of an OpenSim .mot guess and
% write it back out in the same format. Header lines are copied verbatim apart
% from nothing -- row/column counts are unchanged.

fidIn = fopen(inFile,'r');
raw = textscan(fidIn,'%s','Delimiter','\n','Whitespace','');
fclose(fidIn);
lines = raw{1};

iEnd = find(strcmp(strtrim(lines),'endheader'),1);
assert(~isempty(iEnd), 'no endheader in %s', inFile);

% each field is padded with spaces in the file, so trim per label, not just
% the line -- otherwise the pelvis_tx/tz exclusion below silently never matches
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
