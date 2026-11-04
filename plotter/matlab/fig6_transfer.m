%FIG6TRANSFER Transfer results (Fig 6): battery-wise MAPE of seven models
% on the four cross-dataset fine-tuning pairs.
C = ctab();
h = figstyle(190, 66);

V = readtable(fullfile('..','data','fig6_transfer.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
pairs = ["XJTU-2C-to-HUST", "HUST-to-XJTU-2C", "XJTU-2C-to-TJU-0", "TJU-0-to-XJTU-2C"];
plabels = {'XJTU\rightarrowHUST', 'HUST\rightarrowXJTU', 'XJTU\rightarrowTJU', 'TJU\rightarrowXJTU'};
models = ["adaptive","orig","GPR","GRU","LSTM","CNN","MLP"];
mnames = {'Adaptive (ours)', 'PINN4SOH', 'GPR', 'GRU', 'LSTM', 'CNN', 'MLP'};

ax = axes(h); hold(ax, 'on');
w = 0.115;
hb = gobjects(7, 1);
ymax_all = 0;
hb = gobjects(7, 1);
ymax_all = 0;
offs = (-3:3)' * w;
bw = w * 0.82;
for mi = 1:7
    mu = nan(1, 4); sd = nan(1, 4);
    for pi = 1:4
        x = V.MAPE(V.model == models(mi) & V.("case") == pairs(pi)) * 100;
        mu(pi) = mean(x); sd(pi) = std(x);
    end
    hb(mi) = bar(ax, (1:4) + offs(mi), mu, bw, 'FaceColor', C.(models(mi)), ...
        'EdgeColor', 'k', 'LineWidth', 0.4);
    ymax_all = max([ymax_all, max(mu + sd)]);
    errorbar(ax, (1:4) + offs(mi), mu, sd, 'LineStyle', 'none', 'Color', 'k', ...
        'LineWidth', 0.6, 'CapSize', 2);
    if mi == 1
        for pi = 1:4
            text(ax, (1:4) + offs(mi), mu + sd + 0.10, sprintf('%.2f', mu(pi)), ...
                'FontSize', 6.5, 'HorizontalAlignment', 'center', 'FontName', 'Arial');
        end
    end
end
ymax = ymax_all * 1.15;
xlim(ax, [0.45, 4.55]);
xticks(ax, 1:4); xticklabels(ax, plabels);
ylabel(ax, 'MAPE (%)');
set(ax, 'FontSize', 7);
ylim(ax, [0, ymax]);
lg = legend(hb, mnames, 'NumColumns', 7, 'Location', 'northoutside', 'Box', 'off');
lg.ItemTokenSize = [12, 4];

out = fullfile('..', 'fig6_transfer.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
