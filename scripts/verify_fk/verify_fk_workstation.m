% Debugging script: compute ground-truth forward kinematics via
% BioMAC-Sim-Toolbox's own Gait3d_smoothsphere.getFkin, for direct
% comparison against the Python OpenSim-based reconstruction used in
% plot/figure02_v2.py. Throwaway/debug script, not toolbox code.
%
% Loads a single row's q vector (exported from Python, see
% scripts/verify_fk/q_row3_k22.mat: row_idx=3 (2.0 m/s target), raw node
% k=22 -- the exact case already debugged: Python's from-scratch OpenSim FK
% gave calcn_r sphere0 world Y = 0.0994, while the raw state's own algebraic
% contact-point variable yc = 0.0228 for the same physical point. getFkin
% here is meant to say which one (if either) is right.
%
% Model source: uses the EXACT .osim file already used by the Python
% pipeline (not any bundled/generic BioMAC model) -- per explicit user
% instruction, to make this an apples-to-apples comparison.

toolbox_src = '/home/rzlin/ri94mihu/phd/BiomechPriorVAE/BioMAC-Sim-Toolbox/src';
addpath(genpath(toolbox_src));

osim_path = '/home/rzlin/ri94mihu/phd/BiomechPriorVAE/data/model/sipp_generic_runmad_smoothsphere.osim';
% loadMomentArms locates the precomputed *_momentarms.mat via `which(...)`,
% i.e. it searches the MATLAB path, not the .osim file's own directory --
% add it so the existing momentarms file is found instead of recomputing.
addpath(fileparts(osim_path));

fprintf('Instantiating Gait3d_smoothsphere from %s ...\n', osim_path);
model = Gait3d_smoothsphere(osim_path);
fprintf('Model loaded. nDofs=%d\n', model.nDofs);

% DOF names in model order, for sanity-checking against Python's own order
dof_names = model.dofs.Properties.RowNames;
fprintf('DOF order (first 6): %s\n', strjoin(dof_names(1:6), ', '));

% Load q exported from Python, WITH the Python-side coordinate name order
% (do NOT assume MATLAB's own model.dofs order matches -- reorder by name).
data = load('/home/rzlin/ri94mihu/verify_fk/q_row3_k22.mat');
q_py = data.q(:);
py_names = cellstr(data.names);
fprintf('Loaded q (%d values) for row_idx=%d, k=%d\n', length(q_py), data.row_idx, data.k);

if length(q_py) ~= model.nDofs
    error('q length (%d) does not match model.nDofs (%d)', length(q_py), model.nDofs);
end

q = zeros(model.nDofs, 1);
name_mismatch = false;
for i = 1:model.nDofs
    idx = find(strcmp(py_names, dof_names{i}));
    if isempty(idx)
        fprintf('WARNING: MATLAB dof "%s" not found in Python name list -- ORDER MISMATCH, cannot proceed blindly\n', dof_names{i});
        name_mismatch = true;
    else
        q(i) = q_py(idx);
    end
end
if name_mismatch
    error('DOF name mismatch between MATLAB model and Python export -- inspect dof_names vs py_names manually before trusting any q values.');
end
fprintf('q reordered to match MATLAB model.dofs order (name-matched, not position-assumed).\n');

% Call getFkin
FK = model.getFkin(q);

% Extract all segment transforms
seg_names = model.segments.Properties.RowNames;
fprintf('Segments (%d): %s\n', length(seg_names), strjoin(seg_names', ', '));

results = struct();
results.dof_names = dof_names;
results.seg_names = seg_names;
bodies = struct();
for i = 1:length(seg_names)
    name = seg_names{i};
    idxp = (i-2)*12 + (1:3);
    idxR = (i-2)*12 + (4:12);
    if idxp(1) < 1
        continue % ground segment, not in FK array
    end
    p = FK(idxp);
    R = reshape(FK(idxR), 3, 3)'; % row-major per simuMarker.m usage
    safe_name = matlab.lang.makeValidName(name);
    bodies.(safe_name).p = p;
    bodies.(safe_name).R = R;
end
results.bodies = bodies;

% Also dump idxSymmetry for the authoritative INV/OPP table (cross-check
% against the hardcoded table now in figure02_v2.py's RAW_COL). Already
% confirmed by reading update_idxSymmetry's source directly, so this is a
% nice-to-have, not load-bearing -- update_idxSymmetry is protected and may
% not be externally callable; don't let that block the FK result.
results.nDofs = model.nDofs;
try
    sym = model.idxSymmetry; % getter; populated during construction listeners
    results.symmetry_xindex = sym.xindex(1:model.nDofs);
    results.symmetry_xsign = sym.xsign(1:model.nDofs);
catch ME
    fprintf('WARNING: could not read idxSymmetry (%s) -- skipping, already confirmed via source read.\n', ME.message);
end

out_path = '/home/rzlin/ri94mihu/verify_fk/fk_result.mat';
save(out_path, '-struct', 'results');
fprintf('Wrote %s\n', out_path);

% Print calcn_r / toes_r positions directly for quick visual check
fprintf('\ncalcn_r p = [%.4f %.4f %.4f]\n', bodies.calcn_r.p);
fprintf('toes_r  p = [%.4f %.4f %.4f]\n', bodies.toes_r.p);
