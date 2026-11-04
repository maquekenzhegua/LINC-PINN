function violinm(x, pos, w, col)
%VIOLINM Half-scaled violin with mean line (red) and mean+-std caps (black),
% matching the error-bar style of the PINN4SOH paper.
x = x(~isnan(x));
if numel(x) < 2, return; end
[f, xi] = gkde(x, 121);
f = f / max(f) * w;
patch('XData', [pos-f, fliplr(pos+f)], 'YData', [xi, fliplr(xi)], ...
    'FaceColor', col, 'FaceAlpha', 0.40, 'EdgeColor', col, ...
    'LineWidth', 0.7);
m = mean(x); s = std(x);
line([pos-w*0.85, pos+w*0.85], [m, m], 'Color', [0.80 0.08 0.08], 'LineWidth', 1.0);
line([pos, pos], [m-s, m+s], 'Color', 'k', 'LineWidth', 0.7);
line([pos-w*0.5, pos+w*0.5], [m-s, m-s], 'Color', 'k', 'LineWidth', 0.7);
line([pos-w*0.5, pos+w*0.5], [m+s, m+s], 'Color', 'k', 'LineWidth', 0.7);
end
