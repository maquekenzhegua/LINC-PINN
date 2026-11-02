%FIG4MAIN Regular-phase results (Fig 4): (a) per-cycle predicted vs true SOH
% of AdaptiveInvariantPINN for the 11 conditions, colored by absolute error;
% (b) distributions of MAE/MAPE/RMSE over 10 repeats, adaptive vs orig.
C = ctab();
conds = ["XJTU-2C","XJTU-3C","XJTU-R2.5","XJTU-R3","XJTU-RW","XJTU-satellite", ...
         "TJU-0","TJU-1","TJU-2","MIT","HUST"];
ctitles = ["XJTU-2C","XJTU-3C","XJTU-R2.5","XJTU-R3","XJTU-RW","XJTU-satellite","TJU-0","TJU-1","TJU-2","MIT","HUST"];
letters = ['a' repmat(' ',1,11) 'b' repmat(' ',1,10)];

h = figstyle(190, 205);
tl = tiledlayout(h, 6, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---------------- (a) scatter grid, rows 1-3 ----------------
S = readtable(fullfile('..','data','fig4_scatter.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
emax = max(abs(S.pred - S.true));
for k = 1:11
    ax = nexttile(tl, k); hold(ax, 'on');
    R = S(S.condition == conds(k), :);
    e = abs(R.pred - R.true);
    sc = scatter(ax, R.true, R.pred, 2.5, e, 'filled', 'MarkerFaceAlpha', 0.75);
    caxis(ax, [0, emax]);

    lo = min([R.true; R.pred]); hi = max([R.true; R.pred]);
    pad = 0.03 * (hi - lo);
    plot(ax, [lo hi], [lo hi], '--', 'Color', [0.85 0.25 0.20], 'LineWidth', 0.7);
    axis(ax, [lo - pad, hi + pad, lo - pad, hi + pad]);
    title(ax, ctitles(k), 'FontWeight', 'normal', 'FontSize', 6.5);
    if mod(k - 1, 4) == 0, ylabel(ax, 'Predicted SOH', 'FontSize', 7); end
    if k >= 9, xlabel(ax, 'True SOH', 'FontSize', 7); end
    set(ax, 'FontSize', 6);
    ptag(ax, letters(k), -0.13, 1.02);
end
axc = nexttile(tl, 12); hold(axc, 'on');
pc = parula(256);
ybase = 0.02; yspan = 0.93;
for r = 1:64
    ylo = ybase + (r - 1) / 64 * yspan; yhi = ybase + r / 64 * yspan;
    patch(axc, 'XData', [0, 0.55, 0.55, 0], 'YData', [ylo, ylo, yhi, yhi], ...
        'FaceColor', pc(max(1, min(256, round(r / 64 * 256))), :), 'EdgeColor', 'none');
end
axis(axc, 'off');
text(axc, 0.62, ybase, '0', 'FontSize', 6.0, 'HorizontalAlignment', 'left');
text(axc, 0.62, ybase + yspan / 2, sprintf('%.2f', emax / 2), 'FontSize', 6.0, 'HorizontalAlignment', 'left');
text(axc, 0.62, ybase + yspan, sprintf('%.2f', emax), 'FontSize', 6.0, 'HorizontalAlignment', 'left');
text(axc, 0.62, ybase + yspan + 0.10, '|Error|', 'FontSize', 6.2, 'HorizontalAlignment', 'left');


% ---------------- (b) violin grid, rows 4-6 ----------------
V = readtable(fullfile('..','data','fig4_violin.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
metrics = ["MAE","MAPE","RMSE"];
for k = 1:11
    ax = nexttile(tl, 12 + k); hold(ax, 'on');
    R = V(V.("case") == conds(k), :);
    ymax = 0;
    for m = 1:3
        A = 100 * R.(metrics(m))(R.model == "adaptive");
        O = 100 * R.(metrics(m))(R.model == "orig");
        violinm(A, m - 0.22, 0.19, C.adaptive);
        violinm(O, m + 0.22, 0.19, C.orig);
        ymax = max([ymax, max(A) + std(A), max(O) + std(O)]);
    end
    ylim(ax, [0, ymax * 1.25]);
    xlim(ax, [0.45, 3.55]);
    xticks(ax, 1:3); xticklabels(ax, {'MAE','MAPE','RMSE'});
    ax.XAxis.FontSize = 6;
    set(ax, 'FontSize', 6);
    title(ax, ctitles(k), 'FontWeight', 'normal', 'FontSize', 6.5);
    if mod(k - 1, 4) == 0, ylabel(ax, 'Error (%)', 'FontSize', 7); end
    if k >= 9, xlabel(ax, 'Metric', 'FontSize', 7); end
end
% shared legend parked in the free part of layout tile 12
ph = patch('XData', NaN, 'YData', NaN, 'FaceColor', C.adaptive, 'EdgeColor', C.adaptive);
po = patch('XData', NaN, 'YData', NaN, 'FaceColor', C.orig, 'EdgeColor', C.orig);
lg = legend([ph po], {'Adaptive (ours)', 'PINN4SOH'}, 'Box', 'off');
lg.Position = [0.758, 0.045, 0.14, 0.05];

ptag(nexttile(tl, 13), 'b', -0.13, 1.03);
ptag(nexttile(tl, 13), 'b', -0.13, 1.03);
out = fullfile('..', 'fig4_main.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
