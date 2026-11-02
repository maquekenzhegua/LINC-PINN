%FIG7ABLATIONGAMMA (Fig 7) (a) necessity ablation at n=1 on HUST and
% XJTU-2C small-sample tasks; (b) monotonicity-weight sweep: MAPE and
% predicted SOH span versus gamma.
C = ctab();
h = figstyle(190, 66);
tl = tiledlayout(h, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---------------- (a) ablation bars ----------------
ax = nexttile(tl); hold(ax, 'on');
A = readtable(fullfile('..','data','fig7_ablation.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
configs = ["full","no_operator","src_scaler","latent3","no_rowmean","no_anchored"];
clabels = {'Full', 'w/o operator', 'src. scaler', 'latent 8\rightarrow3', ...
           'w/o row-mean', 'w/o anchored'};
cases = ["small-HUST-n1", "small-XJTU-2C-n1"];
casecols = [C.batches(1, :); C.batches(4, :)];

w = 0.36;
bar_handles = gobjects(1, 2);
for ci = 1:2
    mu = nan(1, 6); sd = nan(1, 6);
    for k = 1:6
        if configs(k) == "full"
            x = A.MAPE(A.variant == "full" & A.model == "adaptive" & A.("case") == cases{ci}) * 100;
        elseif configs(k) == "src_scaler"
            x = A.MAPE(A.variant == "src_scaler" & A.("case") == cases{ci}) * 100;
        else
            x = A.MAPE(A.variant == configs(k) & A.("case") == cases{ci}) * 100;
        end
        mu(k) = mean(x); sd(k) = std(x);
    end
    bar_handles(ci) = bar(ax, (1:6) + (ci - 1.5) * w, mu, w * 0.85, 'FaceColor', casecols(ci, :), ...
        'EdgeColor', 'k', 'LineWidth', 0.4);
    errorbar(ax, (1:6) + (ci - 1.5) * w, mu, sd, 'LineStyle', 'none', ...
        'Color', 'k', 'LineWidth', 0.6, 'CapSize', 2);
    % PINN4SOH reference line
    o = A.MAPE(A.variant == "orig" & A.("case") == cases{ci}) * 100;
    yref = mean(o);
    plot(ax, [0.35, 6.65], [yref, yref], '--', 'Color', casecols(ci, :), 'LineWidth', 0.9);
    text(ax, 0.42, yref, sprintf('PINN4SOH %.1f%% ', yref), 'FontSize', 6.8, ...
        'Color', casecols(ci, :), 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom', 'FontName', 'Arial');
end
ylim(ax, [0, 5.9]);
xlim(ax, [0.35, 6.65]);
xticks(ax, 1:6); xticklabels(ax, clabels);
ax.XAxis.FontSize = 6.8;
ylabel(ax, 'MAPE at n=1 (%)');
set(ax, 'FontSize', 6.5);
legend(bar_handles, {'HUST', 'XJTU-2C'}, 'Location', 'northeast', 'Box', 'off');
ptag(ax, 'a', -0.13, 1.06);

% ---------------- (b) gamma sweep ----------------
ax = nexttile(tl); hold(ax, 'on');
G = readtable(fullfile('..','data','fig7_gamma.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
gx = [0.2 1 2 5 10];
yyaxis(ax, 'left');
for n = 1:4
    mu = nan(1, 5); sd = nan(1, 5);
    for gi = 1:5
        x = G.MAPE(G.gamma == gx(gi) & G.n == n) * 100;
        mu(gi) = mean(x); sd(gi) = std(x);
    end
    errorbar(ax, gx, mu, sd, '-o', 'Color', C.blues(n, :), 'LineWidth', 0.9, ...
        'MarkerSize', 3.5, 'MarkerFaceColor', C.blues(n, :), 'CapSize', 2);
end
ylabel(ax, 'MAPE (%)');
ax.YAxis(1).Color = [0.13 0.13 0.13];
ax.YAxis(1).FontSize = 7;
ylim(ax, [0, 11]);
yyaxis(ax, 'right');
mu = nan(1, 5); sd = nan(1, 5);
for gi = 1:5
    x = G.pred_range(G.gamma == gx(gi));
    mu(gi) = mean(x); sd(gi) = std(x);
end
tr = mean(G.true_range);
errorbar(ax, gx, mu, sd, '--s', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.9, ...
    'MarkerSize', 3.5, 'MarkerFaceColor', [0.45 0.45 0.45], 'CapSize', 2);
plot(ax, [0.2 10], [tr tr], ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 0.9);
text(ax, 0.23, tr + 0.012, 'measured span', 'FontSize', 6.8, 'Color', [0.2 0.2 0.2], 'HorizontalAlignment', 'left');
ylabel(ax, 'Predicted SOH span');
ax.YAxis(2).Color = [0.45 0.45 0.45];
ax.YAxis(2).FontSize = 7;
ylim(ax, [0.1, 0.42]);
xlim(ax, [0.18, 11]);
ax.XGrid = 'on'; ax.GridAlpha = 0.18;
set(ax, 'XScale', 'log', 'FontSize', 7);
xticks(ax, [0.2 1 2 5 10]);
xticklabels(ax, {'0.2','1','2','5','10'});
xlabel(ax, 'Monotonicity weight \gamma');
legend({'n=1','n=2','n=3','n=4','span (right)'}, 'Location', 'northwest', 'Box', 'off');
ptag(ax, 'b', -0.13, 1.06);

out = fullfile('..', 'fig7_ablation_gamma.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
