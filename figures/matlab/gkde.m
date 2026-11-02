function [d, xi] = gkde(x, npoints)
%GKDE Gaussian kernel density estimate without the Statistics Toolbox.
% Silverman bandwidth; [d,xi] mirrors the ksdensity calling pattern.
if nargin < 2, npoints = 121; end
x = x(:);
x = x(~isnan(x));
n = numel(x);
if n < 2
    d = [1; 1]; xi = [0; 1]; return
end
sig = std(x);
if sig <= 0, sig = eps; end
h = 1.06 * sig * n^(-1/5);
lo = min(x) - 3 * h;
hi = max(x) + 3 * h;
xi = linspace(lo, hi, npoints)';                 % column, length npoints
z = (xi - x') / h;                               % npoints x n
d = sum(exp(-0.5 * z.^2), 2) / (n * sqrt(2 * pi) * h);
end
