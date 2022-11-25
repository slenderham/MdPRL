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

recovery_results_by_input = load('../files/RPL2Analysis_Attention_model_recovery_by_input_thresh_both_5_250.mat');
recovery_results_by_attn = load('../files/RPL2Analysis_Attention_model_recovery_by_attn_thresh_both_5_250.mat');
parameter_recovery_results = load('../files/RPL2Analysis_Attention_param_recovery_5_500.mat');
% parameter_recovery_results2 = load('../files/RPL2Analysis_Attention_param_recovery_5_250_2.mat');

nSample = 100;
ntrials = 432;

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

all_model_names_legend = ["F", "F+O", "F+C_{untied}", "F+C_{feat attn}", "F+C_{tied}"];
[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];
xlabels = strcat(attn_modes(:,1),"X",attn_modes(:,2));

set(0,'defaultAxesFontSize',14)

%%

for a_fit=1:10
    for a_sim=1:10
        for cnt_samp=1:100
            num_params = length(recovery_results_by_attn.fit_results{5, a_fit, cnt_samp, 5, a_sim}.params);
            BIC_temp_by_attn(a_fit, cnt_samp, a_sim) = ...
                2*recovery_results_by_attn.fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
                log(ntrials)*num_params;
            AIC_temp_by_attn(a_fit, cnt_samp, a_sim) = ...
                2*recovery_results_by_attn.fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
                2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        end
    end
end


for m_fit=1:5
    for m_sim=1:5
        for cnt_samp=1:100
            num_params = length(recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.params);
            BIC_temp_by_input(m_fit, cnt_samp, m_sim) = ...
                2*recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.fval+...
                log(ntrials)*num_params;
            AIC_temp_by_input(m_fit, cnt_samp, m_sim) = ...
                2*recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.fval+...
                2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        end
    end
end

% true_pars = [];
% fit_pars = [];
% for cnt_samp=1:100
%     true_pars = [true_pars; parameter_recovery_results.sim_results{5, 3, cnt_samp}.params];
%     fit_pars = [fit_pars; parameter_recovery_results.fit_results{5, 3, cnt_samp, 5, 3}.params];
% end
% 
% for cnt_samp=1:100
%     true_pars = [true_pars; parameter_recovery_results2.sim_results{5, 3, cnt_samp}.params];
%     fit_pars = [fit_pars; parameter_recovery_results2.fit_results{5, 3, cnt_samp, 5, 3}.params];
% end

for cnt_samp=1:100
    true_pars = [true_pars; recovery_results_by_input.sim_results{5, 3, cnt_samp}.params];
    fit_pars = [fit_pars; recovery_results_by_input.fit_results{5, 3, cnt_samp, 5, 3}.params];
end

for cnt_samp=1:100
    true_pars = [true_pars; recovery_results_by_attn.sim_results{5, 3, cnt_samp}.params];
    fit_pars = [fit_pars; recovery_results_by_attn.fit_results{5, 3, cnt_samp, 5, 3}.params];
end


%%
conf_mat_alpha = zeros(5);
conf_mat_pxp = zeros(5);
for m=1:5
    [alpha, ~, xp, pxp, bor, g] = bms(-BIC_temp_by_input(:,:,m)'/2, mat2cell((1:5)', repmat([1], 1, 5)));
    disp(bor)
    conf_mat_alpha(m,:) = alpha/sum(alpha);
    conf_mat_pxp(m,:) = pxp;
end
figure; 
imagesc(conf_mat_alpha);
caxis([0 1]);
xticks(1:5)
xticklabels(all_model_names_legend)
xlabel('Fit Model')
yticks(1:5)
yticklabels(all_model_names_legend)
ylabel('Simulated Model')
colorbar()
[txs, tys] = meshgrid(1:5, 1:5);
colormap(flipud(bone))
txts = text(txs(:)-0.2, tys(:), string(num2str(conf_mat_pxp(:), '%.2f')), 'FontSize', 18);
for i=1:5
    for j=1:5
        if (conf_mat_alpha(i,j)>0.5)
            txts((j-1)*5+i).Color = [1 1 1];
        end
    end
end

%%
conf_mat_alpha = zeros(10);
conf_mat_pxp = zeros(10);
for a=1:10
    [alpha, ~, xp, pxp, bor, ~] = bms(-BIC_temp_by_attn(:,:,a)'/2, mat2cell((1:10)', repmat([1], 1, 10)));
    disp(bor)
    conf_mat_alpha(a,:) = alpha/sum(alpha);
    conf_mat_pxp(a,:) = pxp;
end
figure;
imagesc(conf_mat_alpha);
xticks(1:10)
xticklabels(xlabels)
xlabel('Fit Model')
yticks(1:10)
yticklabels(xlabels)
ylabel('Simulated Model')
colorbar()
title('Confusion Matrix')
[txs, tys] = meshgrid(1:10, 1:10);
colormap(flipud(bone))
txts = text(txs(:)-0.4, tys(:), string(num2str(conf_mat_pxp(:), '%.2f')), 'FontSize', 14);
for i=1:10
    for j=1:10
        if (conf_mat_alpha(i,j)>0.4)
            txts((j-1)*10+i).Color = [1 1 1];
        end
    end
end

%% parameter recovery
figure
imagesc(corr(true_pars, fit_pars, 'type', 'spearman'))
caxis([-1, 1])
colorbar
param_names = {'bias', '\beta', '\omega', 'd', '\alpha_+', '\alpha_-', '\gamma'};
xticklabels(param_names);
yticklabels(param_names);
colormap bluewhitered
xlabel('True Parameters')
ylabel('Fit Parameters')
[txs, tys] = meshgrid(1:7, 1:7);
[param_corr_mat, ps] = corr(true_pars, fit_pars, 'type', 'spearman');
txts = text(txs(:)-0.4, tys(:), string(num2str(param_corr_mat(:), '%.2f')), 'FontSize', 18);
for i=1:7
    for j=1:7
        if (param_corr_mat(i,j)>0.39)
            txts((j-1)*7+i).Color = [1 1 1];
        end
    end
end

figure
for i=1:7
    subplot(3,3,i)
    if (i==2 || i==7)
        pseudo_log10 = @(x) asinh(x/2)/log(10);
        scatter(pseudo_log10(true_pars(:,i)), pseudo_log10(fit_pars(:,i)+1e-6),[],[0.4, 0.4, 0.4]); hold on
        xlim(max([xlim; ylim]))
        ylim(max([xlim; ylim]))
        plot(xlim, ylim, 'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
        ll = lsline;
        
        xticks(0:2)
        yticks(0:2)
        xticklabels("10^"+xticks);
        yticklabels("10^"+yticks);
    else
        scatter(true_pars(:,i), fit_pars(:,i),[],[0.4, 0.4, 0.4]); hold on
        xlim(max([xlim; ylim]))
        ylim(max([xlim; ylim]))
        plot(xlim, ylim, 'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
        ll = lsline;
    end
    set(ll, 'linewidth', 1, 'color', 'black');

    title(param_names{i})
end