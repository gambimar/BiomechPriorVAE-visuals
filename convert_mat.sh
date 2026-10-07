# `data/model` carries the per-variant cached model files BioMAC resolves off
# the MATLAB path (<variant>.mat + <variant>_momentarms.mat next to the
# .osim). Without it on the path a variant's model cannot be loaded/
# initialized and getMetabolicCost fails with "model was not initialized" --
# so it is added here alongside scripts/, not left to the saved pathdef.
LM_PROJECT=iwse /Applications/MATLAB_R2024b.app/bin/matlab -r "addpath(genpath(what('scripts').path)); addpath(genpath(fullfile(pwd,'data','model'))); convert_mat_to_plain('results_sim/simulations','results_sim/converted'); quit()" -nodesktop -nosplash -nodisplay
