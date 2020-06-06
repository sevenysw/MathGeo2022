%%% Seismic data denoising (2D) using 2D denoising CNN.
%%%
%%%
clear;close all
addpath('seismicData');
addpath('utilities');
addpath('seismicData/masks');

%%% choose data, the default variable in .mat is 'D'
dataChoice = 1;
select = 1;
switch dataChoice
    case 1
        Data = 'hyperbolic-events';
    otherwise
        error('Unexpected choice.');
end
load([Data, '.mat'])
Dataname = Data;


%%% ------------------- Parameters setting -------------------------------
noiseL = 0;                 % noise level, valid range [0, 255]

%%% some other parameters setting for result visualizing and saving out.
dx = 0.01;
dt = 0.004;
freqThresh = 100;
showResult = 1;
useGPU = 0;
showFeatures = 0;
saveFeatures = 0;
%%% -----------------------------------------------------------------------

%%% First of all, We have to cast the orignal data into value
%%% range of [0, 1].
label = single(D);
[m, n, l] = size(label);
%%% normalize data to [0, 1]
xmin = min(label(:));
label = label - xmin;
xmax = max(label(:));
label = label/xmax;

%%% add noise
input = label + single(noiseL/255*randn(size(label)));

SNRinput = CalSNR(D, input*xmax+xmin);
PSNRinput = Psnr(D, input*xmax+xmin);
disp(['Input SNR: ', num2str(SNRinput), ' PSNR: ', num2str(PSNRinput)]);

%%% load pre-trained denoising CNN and do denoising
folderModel = 'models';
load(fullfile(folderModel,'model.mat'));

net = loadmodel(noiseL, CNNdenoiser);
net = vl_simplenn_tidy(net);

if useGPU
    input = gpuArray(input);
end
res    = vl_simplenn(net, input,[],[],'conserveMemory',false,'mode','test');
output = input - res(end).x;

if useGPU
    output = gather(output);
    input  = gather(input);
end

denoisedResult = output*xmax+xmin;

SNRCur = CalSNR(D, denoisedResult);
PSNRCur = Psnr(D, denoisedResult);
disp(['CNN SNR: ', num2str(SNRCur), ' PSNR: ', num2str(PSNRCur)]);

if showResult
    x = (0:m-1)*dx; t = (0:n-1)*dt;
    fig1 = figure(1); set(gcf, 'color', 'white'), set(gcf, 'Position', [100, 100, 900, 700]), colormap(gray);
    sub1 = subplot(221);
    imagesc(x,t,D),  cb1 = colorbar;%setColorbar(sub1, cb1, -0.02, 0.02, 0.01, 0.3);  axis off;
    xlabel('Distance (km)'); ylabel('Time (s)');
    title('Original Data')
    sub2 = subplot(222);
    imagesc(x,t,input*xmax+xmin), cb2 = colorbar; %setColorbar(sub2, cb2, -0.02, 0.02, 0.01, 0.3);  axis off;
    xlabel('Distance (km)'); ylabel('Time (s)');
    title(['Sigma = ', num2str(noiseL), ' SNR ', num2str(SNRinput)])
    sub3 = subplot(223);
    imagesc(x,t,denoisedResult), cb3 = colorbar; %setColorbar(sub3, cb3, -0.02, 0.02, 0.01, 0.3); axis off;
    xlabel('Distance (km)'); ylabel('Time (s)');
    title(['Reconstructed data,', ' SNR ', num2str(SNRCur, '%2.2f'), 'dB'])
    sub4 = subplot(224);
    imagesc(x,t,D-denoisedResult); cb4 = colorbar;%('Xtick', 0:0.1:1); %setColorbar(sub4, cb4, -0.02, 0.02, 0.01, 0.3); axis off;
    xlabel('Distance (km)'); ylabel('Time (s)');
    title('Reconstrunction error')
    drawnow;  
end
    
nrows = 4;
ncols = 4;
if showFeatures
    fig2 = figure(2);
    I2 = displayMultiImages(res(2).x, [m, n], nrows, ncols);
    fig3 = figure(3);
    I3 = displayMultiImages(res(3).x, [m, n], nrows, ncols);
    fig4 = figure(4);
    I4 = displayMultiImages(res(4).x, [m, n], nrows, ncols);
    fig5 = figure(5);
    I5 = displayMultiImages(res(5).x, [m, n], nrows, ncols);
    fig6 = figure(6);
    I6 = displayMultiImages(res(6).x, [m, n], nrows, ncols);
    fig7 = figure(7);
    I7 = displayMultiImages(res(7).x, [m, n], nrows, ncols); 
    fig8 = figure(8);
    I8 = displayMultiImages(res(8).x, [m, n], nrows, ncols); 
    fig9 = figure(9);
    I9 = displayMultiImages(res(9).x, [m, n], nrows, ncols); 
    fig10 = figure(10);
    I10 = displayMultiImages(res(10).x, [m, n], nrows, ncols); 
    fig11 = figure(11);
    I11 = displayMultiImages(res(11).x, [m, n], nrows, ncols); 
    fig12 = figure(12);
    I12 = displayMultiImages(res(12).x, [m, n], nrows, ncols); 
    fig13 = figure(13);
    I13 = displayMultiImages(res(13).x, [m, n], nrows, ncols); 
    if saveFeatures
    print(fig2, ['seismicResult/DENOISE/cnn-feature-2'], '-depsc')
    print(fig2, ['seismicResult/DENOISE/cnn-feature-2'], '-dpng')
    print(fig3, ['seismicResult/DENOISE/cnn-feature-3'], '-depsc')
    print(fig3, ['seismicResult/DENOISE/cnn-feature-3'], '-dpng')
    print(fig4, ['seismicResult/DENOISE/cnn-feature-4'], '-depsc')
    print(fig4, ['seismicResult/DENOISE/cnn-feature-4'], '-dpng')
    print(fig5, ['seismicResult/DENOISE/cnn-feature-5'], '-depsc')
    print(fig5, ['seismicResult/DENOISE/cnn-feature-5'], '-dpng')
    print(fig6, ['seismicResult/DENOISE/cnn-feature-6'], '-depsc')
    print(fig6, ['seismicResult/DENOISE/cnn-feature-6'], '-dpng')
    print(fig7, ['seismicResult/DENOISE/cnn-feature-7'], '-depsc')
    print(fig7, ['seismicResult/DENOISE/cnn-feature-7'], '-dpng') 
    print(fig8, ['seismicResult/DENOISE/cnn-feature-8'], '-depsc')
    print(fig8, ['seismicResult/DENOISE/cnn-feature-8'], '-dpng') 
    print(fig9, ['seismicResult/DENOISE/cnn-feature-9'], '-depsc')
    print(fig9, ['seismicResult/DENOISE/cnn-feature-9'], '-dpng') 
    print(fig10, ['seismicResult/DENOISE/cnn-feature-10'], '-depsc')
    print(fig10, ['seismicResult/DENOISE/cnn-feature-10'], '-dpng') 
    print(fig11, ['seismicResult/DENOISE/cnn-feature-11'], '-depsc')
    print(fig11, ['seismicResult/DENOISE/cnn-feature-11'], '-dpng') 
    print(fig12, ['seismicResult/DENOISE/cnn-feature-12'], '-depsc')
    print(fig12, ['seismicResult/DENOISE/cnn-feature-12'], '-dpng') 
    print(fig13, ['seismicResult/DENOISE/cnn-feature-13'], '-depsc')
    print(fig13, ['seismicResult/DENOISE/cnn-feature-13'], '-dpng') 
    end
end