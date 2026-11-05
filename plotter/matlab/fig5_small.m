%FIG5SMALL Few-shot results (Fig 5): battery-wise MAPE vs number of
% training batteries n, seven models, XJTU-2C and HUST.
C = ctab();
h = figstyle(190, 62);
tl = tiledlayout(h, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

V = readtable(fullfile('..','data','fig5_small.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
models = ["adaptive","orig","XGBoost","GRU","LSTM","CNN","MLP"];
mnames = {'Adaptive (ours)', 'PINN4SOH', 'XGBoost', 'GRU', 'LSTM', 'CNN', 'MLP'};
mk = {'o', 's', '^', 'v', 'd', '<', '>'};
dss = ["XJTU-2C", "HUST"];
dtitles = {'XJTU-2C', 'HUST'};
letters = {'a', 'b'};
hs = gobjects(7, 1);

for p = 1:2
    ax = nexttile(tl, p); hold(ax, 'on');
    for mi = 1:7
        mu = nan(1, 4); sd = nan(1, 4);
        for n = 1:4
            x = V.MAPE(V.model == models(mi) & V.dataset == dss(p) & V.n == n) * 100;
            mu(n) = mean(x); sd(n) = std(x);
        end
        if mi <= 2
            if mi == 1, col = C.adaptive; else, col = C.orig; end
            hs(mi) = errorbar(ax, 1:4, mu, sd, ['-' mk{mi}], ...
                'Color', col, 'LineWidth', 1.1, ...
                'MarkerSize', 4, 'MarkerFaceColor', col, 'CapSize', 2.5);
        else
            hs(mi) = errorbar(ax, 1:4, mu, sd, ['--' mk{mi}], ...
                'Color', C.(models(mi)), 'LineWidth', 0.9, ...
                'MarkerSize', 4, 'CapSize', 2.5);
        end
    end
    if p == 1
        ylim(ax, [0, 5.6]);
    else
        % headroom for the PINN4SOH n=1 error bar (4.73 +/- 1.05); the CNN
        % n=1 mean (14.75) stays off-scale and is annotated instead
        ylim(ax, [0, 7.2]);
        text(ax, 1.12, 6.85, 'CNN: 14.7%', 'FontSize', 6.8, ...
            'Color', [0.2 0.2 0.2], 'FontName', 'Arial');
    end
    xlim(ax, [0.6, 4.4]);
    xticks(ax, 1:4);
    xlabel(ax, 'Number of training batteries n');
    ylabel(ax, 'MAPE (%)');
    title(ax, dtitles{p}, 'FontWeight', 'normal', 'FontSize', 7);
    set(ax, 'FontSize', 7);
    ptag(ax, letters{p}, -0.14, 1.06);
end
lg = legend(hs, mnames, 'NumColumns', 4, 'Box', 'off');
lg.Layout.Tile = 'north';

out = fullfile('..', 'fig5_small.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
