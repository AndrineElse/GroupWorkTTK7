T0 = 3;
fs = 200.0;
T = 1.0/fs;
N = T0/T;

x = linspace(0.0, N*T, N);

y = sin(5 * 2*pi*x) + sin(48 * 2*pi*x)+ sin(50* 2*pi*x);

cwt(y,'bump',fs)

% 
% 
% [imf,residual,info] = emd(y,'Interpolation','pchip');
% 
% nIMF = size(imf);
% nIMF = nIMF(2);
% 
% subplot(nIMF + 2,1,1);
% plot(x,y);
% 
% for n = 1 : nIMF
%     subplot(nIMF + 2,1,n+1);
%     y_temp = imf(:, 1);
%     plot(x,y_temp);
% end
% 
% 
% subplot(nIMF + 2,1,nIMF + 2);
% plot(x,residual);
