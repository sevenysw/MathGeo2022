%%% Generate hyperbolic events data
%%%
%%%
addpath('SeismicLab/codes/synthetics')

dt = 2./1000;
tmax = 2.;
n = 100;
offset = (-n:n)*10;
tau = [.5, .8, 1., 1.4];
v = [1700, 1800, 2000, 2300];
amp = [.4, .4, .6, .5];
f0 = 30;
snr = Inf; 
L = 20;
saveData = 1;
saveFolder = 'seismicData/';
Dataname = 'hyperbolic-events';

D = hyperbolic_events(dt, f0, tmax, offset, tau, v, amp, snr, L);
D = D(:, 1:floor(size(D,2)/4)*4);
figure, imagesc(D), colormap(gray)

if saveData
save([saveFolder, Dataname], 'D');
end
