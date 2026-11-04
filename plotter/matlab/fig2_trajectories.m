%FIG2TRAJECTORIES Degradation trajectories of all 387 batteries (Fig 2).
T = readtable(fullfile('..','data','fig2_trajectories.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
C = ctab();
h = figstyle(190, 108);
tl = tiledlayout(h, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

panels = {'a', 'b', 'c', 'd'};
specs = {  % dataset, batch legend, color source
    {'XJTU', {'2C','3C','R2.5','R3','RW','satellite'}, 'batches'},
    {'TJU',  {'NCA','NCM','NCM+NCA'},                  'tju'},
    {'MIT',  {''},                                     'single'},
    {'HUST', {''},                                     'single'}};

for p = 1:4
    ax = nexttile(tl); hold(ax, 'on');
    ds = specs{p}{1};
    S = T(T.dataset == ds, :);
    if strcmp(specs{p}{3}, 'batches')
        bl = specs{p}{2};
        for b = 1:numel(bl)
            col = C.batches(b, :);
            B = S(S.batch == bl{b}, :);
            u = unique(B.battery);
            for k = 1:numel(u)
                R = sortrows(B(B.battery == u(k), :), 'cycle');
                st = max(1, floor(height(R) / 700));
                plot(ax, R.cycle(1:st:end), R.soh(1:st:end), 'Color', [col 0.50], 'LineWidth', 0.55);
            end
        end
        leg = legend(ax, bl, 'NumColumns', 3, 'FontSize', 6, 'Location', 'northeast', 'Box', 'off');
    elseif strcmp(specs{p}{3}, 'tju')
        bl = specs{p}{2};
        tidx = [3 1 4];
        for b = 1:numel(bl)
            col = C.batches(tidx(b), :);
            B = S(S.batch == bl{b}, :);
            u = unique(B.battery);
            for k = 1:numel(u)
                R = sortrows(B(B.battery == u(k), :), 'cycle');
                st = max(1, floor(height(R) / 700));
                plot(ax, R.cycle(1:st:end), R.soh(1:st:end), 'Color', [col 0.50], 'LineWidth', 0.55);
            end
        end
        leg = legend(ax, bl, 'NumColumns', 3, 'FontSize', 6, 'Location', 'northeast', 'Box', 'off');
    else
        col = C.batches(3, :);
        u = unique(S.battery);
        for k = 1:numel(u)
            R = sortrows(S(S.battery == u(k), :), 'cycle');
            st = max(1, floor(height(R) / 700));
            plot(ax, R.cycle(1:st:end), R.soh(1:st:end), 'Color', [col 0.45], 'LineWidth', 0.55);
        end
    end
    xlim(ax, [0, max(S.cycle)]);
    ylabel(ax, 'SOH');
    if p >= 3, xlabel(ax, 'Cycle'); end
    title(ax, ds, 'FontWeight', 'normal', 'FontSize', 7);
    ptag(ax, panels{p}, -0.10, 1.03);
end
% bottom-row xlabels for top row too? keep only bottom for compactness
nexttile(tl, 1); xlabel('Cycle');
nexttile(tl, 2); xlabel('Cycle');

out = fullfile('..', 'fig2_trajectories.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
