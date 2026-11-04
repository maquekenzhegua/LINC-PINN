%FIG3FEATURES Feature-SOH correlations (a) and invariance of the row
% transform (b): raw vs row-standardized distributions of three features
% across the four datasets.  Heatmap is drawn as patches so the diverging
% colormap never depends on figure-level colormap state.
C = ctab();
h = figstyle(190, 88);
blue = [33 102 172]/255; mid = [247 247 247]/255; red = [178 24 43]/255;

ax0 = axes('Position', [0.075 0.14 0.335 0.78]); hold(ax0, 'on');
R = readtable(fullfile('..','data','fig3_corr.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
dss = ["XJTU","TJU","MIT","HUST"];
M = nan(4, 16);
for i = 1:4
    M(i, :) = R.r(R.dataset == dss(i))';
end
for i = 1:4
    for j = 1:16
        t01 = (M(i, j) + 1) / 2;
        if t01 <= 0.5
            cc_ = blue + (mid - blue) * (2 * t01);
        else
            cc_ = mid + (red - mid) * (2 * t01 - 1);
        end
        patch(ax0, 'XData', j + [-0.5 0.5 0.5 -0.5], 'YData', i + [-0.5 -0.5 0.5 0.5], ...
            'FaceColor', cc_, 'EdgeColor', 'w', 'LineWidth', 0.25);
        tc = 'k';
        if abs(M(i, j)) > 0.55, tc = 'w'; end
        % Numerical values are provided in fig3_corr.csv; color encodes r.
    end
end
xlim(ax0, [0.5, 16.5]); ylim(ax0, [0.5, 4.5]);
set(ax0, 'YDir', 'reverse', 'Box', 'off');
xticks(ax0, 1:16); xticklabels(ax0, compose('%d', 1:16));
xlabel(ax0, 'Feature number');
yticks(ax0, 1:4); yticklabels(ax0, {'XJTU','TJU','MIT','HUST'});
ptag(ax0, 'a', -0.155, 1.035);

% fake colorbar to the right of the heatmap
axb = axes('Position', [0.432, 0.22, 0.010, 0.60]); hold(axb, 'on');
nq = 64;
for q = 1:nq
    tq = -1 + 2 * (q - 0.5) / nq;
    t01 = (tq + 1) / 2;
    if t01 <= 0.5, cq = blue + (mid - blue) * (2 * t01);
    else, cq = mid + (red - mid) * (2 * t01 - 1); end
    patch(axb, 'XData', [0, 1, 1, 0], 'YData', [tq - 1/nq, tq - 1/nq, tq + 1/nq, tq + 1/nq], ...
        'FaceColor', cq, 'EdgeColor', 'none');
end
axis(axb, 'off');
for yy = [-1 -0.5 0 0.5 1]
    plot(axb, [2.15 2.3], [yy yy], 'k-', 'LineWidth', 0.4);
    text(axb, 2.55, yy, sprintf('%.1f', yy), 'FontSize', 6.8, 'FontName', 'Arial', ...
        'HorizontalAlignment', 'left');
end
text(axb, 5.8, 0, 'Pearson r', 'FontSize', 7, 'FontName', 'Arial', ...
    'Rotation', 90, 'HorizontalAlignment', 'center');

% ---- (b) raw vs row-standardized distributions, 3 features x 2 cols ----
V = readtable(fullfile('..','data','fig3_invariance.csv'), 'TextType', 'string', 'VariableNamingRule', 'preserve');
feats = ["v.mean"; "CC Q"; "CV Q"];
funits = {' (V)', ' (Ah)', ' (Ah)'};
x0 = 0.575; wpx = 0.185; gapx = 0.035; rowh = 0.195; rowgap = 0.05;
for r = 1:3
    f = feats(r);
    for cc = 1:2
        tfL = ["raw", "rowstd"];
        tf = tfL(cc);
        xpos = x0 + (cc - 1) * (wpx + gapx);
        ypos = 0.635 - (r - 1) * (rowh + rowgap);
        ax = axes('Position', [xpos, ypos, wpx, rowh]); hold(ax, 'on');
        for i = 1:4
            x = V.value(V.dataset == dss(i) & V.feature == f & V.transform == tf);
            [d, xi] = gkde(x(:), 161);
            plot(ax, xi, d, 'Color', C.datasets(i, :), 'LineWidth', 0.9);
        end
        if cc == 1
            xr = V.value(V.feature == f & V.transform == "raw");
            xlim(ax, [min(xr) - 0.08, max(xr) + 0.08]);
        else
            zr = V.value(V.feature == f & V.transform == "rowstd");
            margin = 0.05 * (max(zr) - min(zr));
            xlim(ax, [min(zr) - margin, max(zr) + margin]);
        end
        set(ax, 'YTick', []);
        if r == 1
            if cc == 1, title(ax, 'raw feature', 'FontWeight', 'normal', 'FontSize', 7);
            else,       title(ax, 'row-standardized', 'FontWeight', 'normal', 'FontSize', 7); end
        end
        if r == 3
            if cc == 1, xlabel(ax, ['value' funits{r}], 'FontSize', 6.5);
            else, xlabel(ax, 'dimensionless value', 'FontSize', 6.5); end
        end
        if cc == 1
            text(ax, 'Units', 'normalized', 'Position', [-0.42, 0.5, 0], ...
                'String', char(f), 'Rotation', 90, 'FontSize', 6.5, ...
                'HorizontalAlignment', 'center', 'Clipping', 'off');
        end
        yl = ylim(ax);
        ylim(ax, [0, yl(2) * 1.15]);
        if r == 2 && cc == 2
            legend(ax, {'XJTU','TJU','MIT','HUST'}, 'NumColumns', 4, ...
                'FontSize', 6.5, 'Position', [0.565 0.89 0.41 0.065], 'Box', 'off');
        end
    end
end
annotation(h, 'textbox', [0.508 0.875 0.02 0.04], 'String', 'b', ...
    'FontSize', 9, 'FontWeight', 'bold', 'EdgeColor', 'none');

out = fullfile('..', 'fig3_features.pdf');
exportgraphics(h, out, 'ContentType', 'vector');
fprintf('wrote %s\n', out);
