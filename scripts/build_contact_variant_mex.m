function build_contact_variant_mex()
% build_contact_variant_mex Compile Mac-native MEX for the contact-stiffness/
% damping variant Gait3d models (softdamp2/4, stiffer2/4).
%
% These variants' classdefs + contact-code generator scripts were fetched
% from the lab workstation (only Linux .mexa64 existed there -- useless on
% this Mac) into BioMAC-Sim-Toolbox/src/model/gait3d/. Their own
% getMexFiles() already: (1) runs the SymPy generator for the small
% contact-specific C file if it isn't already there, (2) `mex -c` compiles
% just that small file, (3) links it against the ALREADY-PRESENT Mac-native
% gait3d_pelvis213_{al,NoDer_al,FK_al}.o and the shared
% gait3d_pelvis213_smoothsphere.c (same for every contact variant, so not
% regenerated) into the final variant MEX -- so this only recompiles the
% small per-variant contact object, not the whole multibody dynamics.
%
% Usage: matlab -batch "run('scripts/build_contact_variant_mex.m')"

toolboxGait3d = '/Users/markusgambietz/PhD/00_MatLab_Projects/BioMAC-Sim-Toolbox/src/model/gait3d';
addpath(genpath(fileparts(toolboxGait3d)));

% One of Gait3d.getMexFiles's recognized OpenSim model-name strings that
% maps to name_MEX='gait3d_pelvis213' (see @Gait3d/getMexFiles.m's switch) --
% picked over the raw .osim filename since getMexFiles keys off the
% *OpenSim model name* embedded in the .osim XML, not the filename.
osimName = 'Running Model For Motions In All Directions';

variants = {'Gait3d_smoothsphere_softdamp2', 'Gait3d_smoothsphere_softdamp4', ...
            'Gait3d_smoothsphere_stiffer2', 'Gait3d_smoothsphere_stiffer4', ...
            'Gait3d_smoothsphere_stiff10x', 'Gait3d_smoothsphere_nodamp'};

for i = 1:numel(variants)
    className = variants{i};
    fprintf('\n=== Building %s ===\n', className);
    try
        nameMEX = feval([className '.getMexFiles'], osimName);
        fprintf('Built: %s\n', nameMEX);
    catch e
        fprintf('FAILED to build %s: %s\n', className, e.message);
    end
end

end
