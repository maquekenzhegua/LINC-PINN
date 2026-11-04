function c = ctab(name)
%CTAB Okabe-Ito derived publication palette (struct of RGB rows).
c.adaptive = [0.835 0.369 0.000];   % vermilion  (ours)
c.orig     = [0.360 0.360 0.360];   % dark gray
c.GPR      = [0.000 0.620 0.451];   % bluish green
c.GRU      = [0.800 0.475 0.655];   % reddish purple
c.LSTM     = [0.906 0.624 0.000];   % orange
c.CNN      = [0.337 0.706 0.914];   % sky blue
c.MLP      = [0.000 0.447 0.698];   % blue
c.batches  = [0.000 0.447 0.698;    % XJTU batches 1-6
              0.906 0.624 0.000;
              0.000 0.620 0.451;
              0.835 0.369 0.000;
              0.800 0.475 0.655;
              0.337 0.706 0.914];
c.datasets = [0.000 0.447 0.698;    % XJTU / TJU / MIT / HUST
              0.835 0.369 0.000;
              0.000 0.620 0.451;
              0.800 0.475 0.655];
c.blues    = [0.620 0.792 0.882;    % n = 1..4 gamma sweep
              0.263 0.576 0.776;
              0.129 0.443 0.710;
              0.031 0.271 0.580];
end
