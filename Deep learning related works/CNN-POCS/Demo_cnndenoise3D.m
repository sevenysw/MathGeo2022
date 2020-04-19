%%% Seismic data denoising (3D) using 2D denoising CNN
%%%
%%%
clear; close all
addpath('seismicData');
addpath('utilities');
addpath('seismicData/masks');
addpath('seismicPlots');

% choose data, the default variable in .mat is 'D'
dataChoice = 1;
switch dataChoice
    case 1
        Data = 'X3Dsyn';
    case 2
        Data = '3Dsbb';
    otherwise
        error('Unexpected choice.');
end
load([Data, '.mat'])
Dataname = Data;

%%% ------------------- Parameters setting -------------------------------
noiseL = 20;                 % noise level, valid range [0, 255]

%%% some other parameters setting for result visualizing and saving out.
saveFolder = '';        % path for saving out results
dx = 0.01;
dt = 0.004;
freqThresh = 50;
showResult = 1;
useGPU = 0;
saveResult = 0;

%%% -----------------------------------------------------------------------

%%% load pre-trained denoisng CNN
folderModel = 'models';
load(fullfile(folderModel,'model.mat'));
net = loadmodel(noiseL, CNNdenoiser);
net = vl_simplenn_tidy(net);

[m, n, l] = size(D);

label = single(D);
%%% normalize data to [0, 1]
xmin = min(label(:));
label = label - xmin;
xmax = max(label(:));
label = label/xmax;

noisyData = label + single(noiseL/255*randn(size(label)));

avgSNRnoisy = CalSNR(D, noisyData*xmax+xmin);
disp(['Noisy SNR: ', num2str(avgSNRnoisy)]);

input = reshape(noisyData, [m, n, 1, l]);
%%% load to GPU
if useGPU
    input = gpuArray(input);
end
res    = vl_simplenn(net, input,[],[],'conserveMemory',true,'mode','test');
output = input - res(end).x;
%%% load back to CPU
if useGPU
    output = gather(output);
    input  = gather(input);
end
denoisedData = squeeze(output)*xmax+xmin;
avgSNRrecon = CalSNR(D, denoisedData);
disp(['Recon SNR: ', num2str(avgSNRrecon)]);

fig = seishow3D(denoisedData, 'dx', dx, 'dt', dt, 'dy', dx, 'colorbar', true, 'colormap', 'gray');

if saveResult
    noisyData = noisyData*xmax + xmin;
    save([saveFolder, Dataname, '_noisy_', num2str(noiseL), '.mat'], 'noisyData');
    save([saveFolder, Dataname, '_imageCNN_', num2str(noiseL), '.mat'], 'denoisedData');
end