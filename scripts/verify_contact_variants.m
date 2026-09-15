function verify_contact_variants()
% verify_contact_variants Numerically confirm each compiled contact-law
% variant actually implements its INTENDED parameters -- not by trusting a
% Python reconstruction against noisy OCP trajectories (as an earlier,
% cruder pass did), but by calling each variant's own compiled contact_al()
% directly, in isolation, with synthetic (yc, ycdot, xcdot, zcdot) inputs via
% a tiny MEX test harness (test_contact_harness.c) that bypasses the entire
% multibody model. The harness returns f[3] = 0 - F_predicted_by_compiled_law,
% so -f(1) (MATLAB 1-indexed) IS the compiled law's own force output directly
% -- no root-finding, no finite-differenced velocities, no OCP solve needed.
%
% Each variant's expected force is computed here from the exact closed-form
% (matching generate_contact_3d.py / contact_3d.al bit-for-bit, including the
% smoothing epsilon and non-negativity guard), so a near-zero residual
% confirms BOTH that stiffness_scale/dissipation_val took effect AND that
% the guard/smoothing behaves as intended.
%
% Usage: matlab -batch "run('scripts/verify_contact_variants.m')"

contactDir = '/Users/markusgambietz/PhD/00_MatLab_Projects/BioMAC-Sim-Toolbox/src/model/gait3d/c_files/contact';
cd(contactDir);

variants = struct( ...
    'name',        {'linear (bare SIPP)', 'smoothsphere (base)', 'softdamp2', 'softdamp4', 'stiffer2', 'stiffer4', 'stiff10x', 'nodamp'}, ...
    'obj',         {'contact_3d_al.o', 'contact_3d_smoothsphere_al.o', 'contact_3d_smoothsphere_softdamp2_al.o', ...
                    'contact_3d_smoothsphere_softdamp4_al.o', 'contact_3d_smoothsphere_stiffer2_al.o', ...
                    'contact_3d_smoothsphere_stiffer4_al.o', 'contact_3d_smoothsphere_stiff10x_al.o', 'contact_3d_smoothsphere_nodamp_al.o'}, ...
    'law',         {'linear', 'hertz', 'hertz', 'hertz', 'hertz', 'hertz', 'hertz', 'hertz'}, ...
    'stiffness',   {NaN, 1.0, 1.0, 1.0, 2.0, 4.0, 10.0, 1.0}, ...
    'dissipation', {NaN, 2.0, 1.0, 0.5, 2.0, 2.0, 2.0, 1e-6});

% Probe points: (yc, ycdot) pairs spanning typical stance depths/rates, plus
% one rapid-liftoff point (large positive ycdot) to also exercise the
% non-negativity guard.
probes = [ ...
    -0.010, -0.05;   %#ok<*NBRACE> % linear-model style probe: moderate compression
    -0.005,  0.10;
     0.015, -0.05;   % hertz-model style probe (yc near/under radius=0.032): compression
     0.025,  0.20;   % hertz-model: shallow penetration, unloading
     0.030,  2.00;   % hertz-model: near liftoff, fast -- exercises the guard
];

fprintf('%-22s %10s %10s %10s %12s\n', 'variant', 'yc', 'ycdot', 'F_compiled', 'F_expected');
for v = 1:numel(variants)
    var = variants(v);
    mexName = sprintf('test_harness_%d', v);
    mex('-largeArrayDims', 'test_contact_harness.c', var.obj, '-output', mexName);
    testFn = str2func(mexName);

    for pidx = 1:size(probes, 1)
        yc = probes(pidx, 1);
        ycdot = probes(pidx, 2);
        f = testFn(yc, ycdot, 0, 0);
        F_compiled = -f(4);  % f(4) is MATLAB-1-indexed f[3] (0-indexed) = the Fy residual at Fx=Fy=Fz=0

        if strcmp(var.law, 'linear')
            F_expected = expected_linear(yc, ycdot);
        else
            F_expected = expected_hertz(yc, ycdot, var.stiffness, var.dissipation);
        end

        err = abs(F_compiled - F_expected);
        flag = '';
        if err > 1e-6 * max(1, abs(F_expected))
            flag = '  <-- MISMATCH';
        end
        fprintf('%-22s %10.4f %10.4f %10.5f %12.5f%s\n', var.name, yc, ycdot, F_compiled, F_expected, flag);
    end
    fprintf('\n');
end

end


function F = expected_linear(yc, ycdot)
% Exact bare-SIPP linear law, contact_3d.al: "stiffness 100 BW per meter"
%   f4 = Fy + 100*0.5*(yc-sqrt(yc^2+0.001^2))*(1-0.75*ycdot)  =>  Fy = -that term
D = yc - sqrt(yc^2 + 0.001^2);
F = -100 * 0.5 * D * (1 - 0.75 * ycdot);
end


function F = expected_hertz(yc, ycdot, stiffness_scale, dissipation_val)
% Exact SmoothSphere Hertz+Hunt-Crossley law, generate_contact_3d.py.
radius = 0.032;
eps_y = 0.001;
stiffness_val = 1.0e6;
bv = 50.0;

a = radius - yc;
d = 0.5 * (a + sqrt(a^2 + eps_y^2));   % smooth max(a,0)
vn = -ycdot;

k_hertz = 0.5 * stiffness_val^(2/3);
C_hertz = stiffness_scale * (4/3) * k_hertz * sqrt(radius * k_hertz);

fh = C_hertz * d^1.5;
fhc = fh * (1.0 + 1.5 * dissipation_val * vn);

guard_zero_vn = -1.0 / (1.5 * dissipation_val);
fhc = fhc * 0.5 * (1.0 + tanh(bv * (vn - guard_zero_vn)));

bw = 1.0;  % matches the harness's get_bodyweight() stub
F = fhc / bw;
end
