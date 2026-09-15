% run_free_speed.m
%
% PredSim with the gait speed left FREE inside [0.6, 2.0] m/s, so the speed
% is predicted rather than imposed. This is the PredSim counterpart of the
% `benchmarks/free/` sets for the three generative models, where nothing was
% commanded and the emergent speed is the quantity of interest.
%
% Why this is well posed: OCP_formulation minimises Jall_sc = sum(Jall)/
% dist_trav_tot, i.e. cost per unit DISTANCE, not per unit time. Freeing the
% speed therefore selects the speed of lowest cost of transport. Had the cost
% been per unit time, the optimiser would simply stand still.
%
% Requires the S.bounds.forward_velocity settings added to
% getDefaultSettings.m and OCP_formulation.m.
%
% The NLP is non-convex and the emergent speed is the whole result, so it is
% solved from THREE initial guesses spanning the interval. If they agree, the
% optimum is a property of the model; if they disagree, the result is a local
% minimum and must be reported as such.

predsim_env();
pathRepo = fileparts(mfilename('fullpath'));
addpath(pathRepo);                              % opensimAD cd's away mid-run
addpath(fullfile(pathRepo,'DefaultSettings'));

resultsRoot = '/home/rzlin/ri94mihu/phd/predsim/results/free_speed';
if ~isfolder(resultsRoot); mkdir(resultsRoot); end

V_LOWER = 0.6;
V_UPPER = 2.0;

% Initial guesses spanning the interval. S.misc.forward_velocity no longer
% constrains anything here -- it only seeds guess.tf and guess.Qdots (see
% getGuess_QR_opti) -- so it is purely the starting point.
v_guess = [0.8 1.25 1.8];
tags    = {'ig080','ig125','ig180'};

% ---- bound relaxation, and why it is necessary ----
% Default_Coordinate_Bounds.csv caps INSTANTANEOUS pelvis_tx velocity at
% 2.73 m/s. Forward velocity oscillates within the stride by ~25-30 % of its
% mean, so a mean near the 2.0 m/s upper bound would push the peak into that
% cap. A binding cap would depress the emergent speed and the result would be
% an artefact of the bound rather than of the cost. Raised to 4.0 m/s so it is
% non-binding across the whole interval; postprocessing must confirm the peak
% stays clear of it.
PELVIS_TX_QDOT = [0, 4.0];    % m/s  (default [0.0278 2.7304])
%
% The pelvis_tilt / pelvis_ty relaxations used in run_baseline_afschrift_speeds
% are deliberately NOT applied: those exist to permit running flight phases,
% and imposing them here would change the walking solution away from the
% settings the rest of the project's walking work uses.

fid = fopen(fullfile(resultsRoot,'run_notes.txt'),'w');
fprintf(fid,'Free-speed PredSim: average velocity constrained to [%g, %g] m/s\n', ...
    V_LOWER, V_UPPER);
fprintf(fid,'instead of the usual equality at S.misc.forward_velocity.\n\n');
fprintf(fid,'Cost is per unit distance (Jall_sc), so the optimum is the\n');
fprintf(fid,'cost-of-transport-optimal speed.\n\n');
fprintf(fid,'Solved from %d initial guesses: %s m/s\n', numel(v_guess), mat2str(v_guess));
fprintf(fid,'Bound override: S.bounds.Qdots pelvis_tx = [%g %g] m/s (default [0.0278 2.7304])\n', ...
    PELVIS_TX_QDOT);
fprintf(fid,'Otherwise the Afschrift et al. settings used for the speed baselines:\n');
fprintf(fid,'Falisse_et_al_2022 model, mtp_type 2022paper, bounds.a.lower 0.01,\n');
fprintf(fid,'N_meshes 50, N_threads 2.\n');
fclose(fid);

timingFile = fullfile(resultsRoot,'timings.csv');
if ~isfile(timingFile)
    fid = fopen(timingFile,'w');
    fprintf(fid,['v_guess,wallclock_s,wallclock_min,success,return_status,' ...
                 'iterations,objective,vel_emergent_mps,savename\n']);
    fclose(fid);
end

for i = 1:numel(v_guess)
    v = v_guess(i);

    cd(pathRepo);   % opensimAD leaves cwd inside its build tree
    [S] = initializeSettings('Falisse_et_al_2022');
    S.subject.name          = 'Falisse_et_al_2022';
    S.misc.forward_velocity = v;        % initial guess only, not a constraint
    S.misc.save_folder      = fullfile(resultsRoot, tags{i});

    % ---- the actual experiment ----
    S.bounds.forward_velocity.lower = V_LOWER;
    S.bounds.forward_velocity.upper = V_UPPER;

    % ---- settings shared with the speed baselines ----
    S.bounds.a.lower          = 0.01;
    S.solver.N_meshes         = 50;
    S.solver.N_threads        = 2;
    S.solver.run_as_batch_job = false;

    S.bounds.Qdots = {'pelvis_tx', PELVIS_TX_QDOT(1), PELVIS_TX_QDOT(2)};

    % every guess starts from the shipped walking IG, so the three runs differ
    % only in the seeded speed and stay independent of each other
    S.solver.IG_selection = fullfile(pathRepo,'OCP','IK_Guess_Full_GC.mot');
    S.solver.IG_selection_gaitCyclePercent = 100;

    osim_path = fullfile(pathRepo,'Subjects',S.subject.name,[S.subject.name '.osim']);

    fprintf('\n=== FREE SPEED, GUESS %.2f m/s START %s ===\n', v, datestr(now));
    ok = true; savename = '';
    t0 = tic;
    try
        savename = runPredSim(S, osim_path);
    catch err
        ok = false;
        fprintf('!!! RUN FAILED for guess %.2f m/s: %s\n', v, err.message);
    end
    el = toc(t0);
    fprintf('=== GUESS %.2f m/s DONE in %.1f s (%.2f min) ===\n', v, el, el/60);

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
    fprintf(fid,'%.2f,%.1f,%.3f,%g,%s,%g,%g,%g,%s\n', ...
        v, el, el/60, success, retstat, iters, objv, velach, savename);
    fclose(fid);

    fprintf('--> emergent speed from guess %.2f: %.4f m/s\n', v, velach);
end

fprintf('\n=== FREE-SPEED RUNS COMPLETE. Timings: %s ===\n', timingFile);
