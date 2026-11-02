function ptag(ax, letter, dx, dy)
%PTAG Bold panel letter at the top-left of an axes (axes-normalized coords).
if nargin < 3, dx = -0.16; end
if nargin < 4, dy = 1.05; end
text(ax,'Units','normalized','Position',[dx dy 0],'String',letter, ...
    'FontSize',9,'FontWeight','bold','Clipping','off','Interpreter','tex');
end
