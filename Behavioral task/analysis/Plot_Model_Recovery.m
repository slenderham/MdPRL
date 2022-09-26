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

recovery_results_by_input = load('../files/RPL2Analysis_Attention_model_recovery_by_input.mat');
recovery_results_by_attn = load('../files/RPL2Analysis_Attention_model_recovery_by_attn.mat');

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
            %             AIC_temp(a_fit, cnt_samp, a_sim) = ...
            %                 2*fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
            %                 2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
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
            %             AIC_temp(a_fit, cnt_samp, a_sim) = ...
            %                 2*fit_results{5, a_fit, cnt_samp, 5, a_sim}.fval+...
            %                 2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        end
    end
end

for cnt_samp=1:100
    true_pars(cnt_samp,:) = recovery_results_by_input.sim_results{5, 3, cnt_samp}.params;
    fit_pars(cnt_samp,:) = recovery_results_by_input.fit_results{5, 3, cnt_samp, 5, 3}.params;
end

for cnt_samp=1:100
    true_pars(cnt_samp+100,:) = recovery_results_by_attn.sim_results{5, 3, cnt_samp}.params;
    fit_pars(cnt_samp+100,:) = recovery_results_by_attn.fit_results{5, 3, cnt_samp, 5, 3}.params;
end


%%
conf_mat_alpha = zeros(5);
conf_mat_pxp = zeros(5);
for m=1:5
    [alpha, ~, xp, pxp, bor, ~] = bms(-BIC_temp_by_input(:,:,m)'/2, mat2cell((1:5)', repmat([1], 1, 5)));
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
txts = text(txs(:)-0.2, tys(:), string(num2str(conf_mat_pxp(:), '%.2f')), 'FontSize', 14);
for i=1:10
    for j=1:10
        if (conf_mat_alpha(i,j)>0.5)
            txts((j-1)*10+i).Color = [1 1 1];
        end
    end
end

%% parameter recovery
figure
imagesc(corr(true_pars, fit_pars, 'type', 'kendall'))
caxis([-1, 1])
colorbar
param_names = {'bias', '\beta', '\omega', 'd', '\alpha_+', '\alpha_-', '\gamma'};
xticklabels(param_names);
yticklabels(param_names);
colormap winter
xlabel('True Parameters')
ylabel('Fit Parameters')
[txs, tys] = meshgrid(1:7, 1:7);
[param_corr_mat, ps] = corr(true_pars, fit_pars, 'type', 'kendall');
txts = text(txs(:)-0.4, tys(:), string(num2str(param_corr_mat(:), '%.2f')), 'FontSize', 18);
for i=1:7
    for j=1:7
        if (param_corr_mat(i,j)<0.4)
            txts((i-1)*7+j).Color = [1 1 1];
        end
    end
end

figure
for i=1:7
    subplot(3,3,i)
    if (i==2 || i==7)
        pseudo_log10 = @(x) asinh(x/2)/log(10);
        scatter(pseudo_log10(true_pars(:,i)), pseudo_log10(fit_pars(:,i)+1e-6)); hold on
        ll = lsline;
        xlim([min(pseudo_log10(true_pars(:,i))) max(pseudo_log10(true_pars(:,i)))])
        ylim([min(pseudo_log10(true_pars(:,i))) max(pseudo_log10(true_pars(:,i)))])
        plot([min(pseudo_log10(true_pars(:,i))) max(pseudo_log10(true_pars(:,i)))], ...
            [min(pseudo_log10(true_pars(:,i))) max(pseudo_log10(true_pars(:,i)))], ...
            'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
        xticks(0:2)
        yticks(0:2)
        xticklabels("10^"+xticks);
        yticklabels("10^"+yticks);
    else
        scatter(true_pars(:,i), fit_pars(:,i)); hold on
        ll = lsline;
        xlim([min(true_pars(:,i)) max(true_pars(:,i))])
        ylim([min(true_pars(:,i)) max(true_pars(:,i))])
        plot([min(true_pars(:,i)) max(true_pars(:,i))], ...
             [min(true_pars(:,i)) max(true_pars(:,i))], ...
             'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
%         xticks(-1:0.2:1)
%         yticks(-1:0.2:1)
    end
    set(ll, 'linewidth', 1, 'color', 'black');

    title(param_names{i})
end