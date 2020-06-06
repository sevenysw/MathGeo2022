clear;

addpath('utilities');
addpath('seismicPlots');
addpath('seismicData');
% choose data
dataChoice = 1;
switch dataChoice
    case 1
        Data = 'hyperbolic-events';
    otherwise
        error('Unexpected choice.');
end
load([Data, '.mat'])
if strcmp(Data, '7blocks')
 D = D{select};
end

sampleType = 'randc';
Ratio= .3;
saveMask = 1;

[m, n] = size(D);
mask = projMask(D, Ratio, sampleType);

fig1 = figure(1); 
set(gcf, 'color', 'white'), set(gcf, 'Position', [100, 100, 900, 700]) 
imagesc(mask.*D);
colormap(seismic(2));

if saveMask
    save(['seismicData/masks/','mask',num2str(m),'x',num2str(n),sampleType,num2str(floor(Ratio*100)),'.mat'], 'mask');
end