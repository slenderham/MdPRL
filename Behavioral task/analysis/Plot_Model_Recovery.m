clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_sim/")
addpath("../utils")
addpath("../files/")
addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
addpath("../models/")
addpath("../../PRLexpv3_5v2/")

load('../files/RPL2Analysis_Attention_model_recovery0.mat')

nSample = 100;
ntrials = 432;

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];
xlabels = strcat(attn_modes(:,1),"X",attn_modes(:,2));

set(0,'defaultAxesFontSize',14)

%%

sim_dim = 5;

for a_fit=1:10
    for a_sim=1:10
        for cnt_samp=1:100
            BIC_temp(a_fit, cnt_samp, a_sim) = ...
                2*fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
                log(ntrials)*length(fit_results{5, a_fit, cnt_samp, 5, a_sim}.params);
            AIC_temp(a_fit, cnt_samp, a_sim) = ...
                2*fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
                2*length(fit_results{5, a_fit, cnt_samp, 5, a_sim}.params);
        end
    end
end

for a=1:10
    for cnt_samp=1:100
        true_pars{a}(cnt_samp,:) = sim_results{5, a, cnt_samp}.params;
        fit_pars{a}(cnt_samp,:) = fit_results{5, a, cnt_samp, 5, a}.params;
    end
end


%% 
conf_mat = zeros(10);
for a=1:10
    [alpha, ~, xp, pxp, bor, ~] = bms(-BIC_temp(:,:,a)'/2, mat2cell((1:10)', repmat([1], 1, 10)));
    disp(bor)
    conf_mat(a,:) = pxp;
end

figure;
imagesc(conf_mat);
xticks(1:10)
xticklabels(xlabels)
% xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xlabel('Fit Model')
yticks(1:10)
yticklabels(xlabels)
% yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
ylabel('Simulated Model')
colorbar()
title('Confusion Matrix')
[txs, tys] = meshgrid(1:10, 1:10);
text(txs(:)-0.2, tys(:), string(num2str(conf_mat(:), '%.2f')))

%%
% for s=1:nSample
%     for m=1:2
%         for n=1:2
%             [M, I] = min(permute(ll_temp(m,n,s,:,:),[1 2 3 5 4]), [],[4 5],'linear');
%             inv_mat(I,(m-1)*2+n) = inv_mat(I,(m-1)*2+n)+1/nSample;
%         end
%     end
% end

inv_mat = zeros(10);
for a=1:10
    [alpha, ~, xp, pxp, bor, ~] = bms(-squeeze(BIC_temp(a,:,:))/2, mat2cell((1:10)', repmat([1], 1, 10)));
    disp(bor)
    inv_mat(:,a) = pxp;
end

figure;
imagesc(inv_mat);
xticks(1:10)
xticklabels(xlabels)
% xticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
xlabel('Fit Model')
yticks(1:10)
yticklabels(xlabels)
% yticklabels({'diff X diff', 'diff X sum', 'sum X diff', 'sum X sum'})
ylabel('Simulated Model')
colorbar()
title('Confusion Matrix')
[txs, tys] = meshgrid(1:10, 1:10);
text(txs(:)-0.2, tys(:), string(num2str(inv_mat(:), '%.2f')))

%% 

