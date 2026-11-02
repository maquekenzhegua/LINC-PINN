%RUN_ALL Generate every MATLAB data figure (Fig 2-7).
cd(fileparts(mfilename('fullpath')));
fig2_trajectories;
fig3_features;
fig4_main;
fig5_small;
fig6_transfer;
fig7_ablation_gamma;
fprintf('run_all: all figures done.\n');
