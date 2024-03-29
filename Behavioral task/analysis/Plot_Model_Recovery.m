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

%%

recovery_results_by_input = load('../files/RPL2Analysis_Attention_model_recovery_by_input_thresh_both.mat');
recovery_results_by_attn = load('../files/RPL2Analysis_Attention_model_recovery_by_attn_thresh_both.mat');
parameter_recovery_results = load('../files/RPL2Analysis_Attention_param_recovery_1_500.mat');

nSample = 100;
ntrials = 432;

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

all_model_names_legend = ["F", "F+O", "F+C_{separate}", "F+C_{feat attn}", "F+C_{spread}", "F+C_{joint}"];
all_model_names_legend = all_model_names_legend([1 2 3 4 6]);
[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];
xlabels = strcat(attn_modes(:,1),"X",attn_modes(:,2));

set(0,'defaultAxesFontSize',14)

%%

aaa = 6;

for a_fit=1:10
    % for a_fit=1:length(attn_modes)
    for a_sim=1:10
        %     for a_sim=1:length(attn_modes)
        for cnt_samp=1:100
            true_ll(a_sim, cnt_samp) = recovery_results_by_attn.sim_results{aaa, a_sim, cnt_samp}.fval;
            fit_ll(a_fit, a_sim, cnt_samp) = recovery_results_by_attn.fit_results{aaa, a_fit, cnt_samp, aaa, a_sim}.fval;
            rec_params{a_fit, a_sim}(cnt_samp, :) = recovery_results_by_attn.fit_results{aaa, a_fit, cnt_samp, aaa, a_sim}.params;
            sim_params{a_sim}(cnt_samp, :) = recovery_results_by_attn.sim_results{aaa, a_sim, cnt_samp}.params;
            num_params = length(recovery_results_by_attn.fit_results{aaa, a_fit, cnt_samp, aaa, a_sim}.params);
            BIC_temp_by_attn(a_fit, cnt_samp, a_sim) = ...
                2*recovery_results_by_attn.fit_results{aaa, a_fit, cnt_samp, aaa, a_sim}.fval+log(ntrials)*num_params;
            %             AIC_temp_by_attn(a_fit, cnt_samp, a_sim) = ...
            %                 2*recovery_results_by_attn.fit_results{6, a_fit, cnt_samp, 6, a_sim}.fval+...
            %                 2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        end
    end
end

for m_fit=1:6
    for m_sim=1:6
        %     for m_sim=1:5
        for cnt_samp=1:100
%             true_ll(a_sim, cnt_samp) = recovery_results_by_attn.sim_results{aaa, a_sim, cnt_samp}.fval;
%             params{a_sim}(cnt_samp, :) = recovery_results_by_attn.sim_results{aaa, a_sim, cnt_samp}.params;
            num_params = length(recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.params);
            BIC_temp_by_input(m_fit, cnt_samp, m_sim) = ...
                2*recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.fval+...
                log(ntrials)*num_params;
            %             AIC_temp_by_input(m_fit, cnt_samp, m_sim) = ...
            %                 2*recovery_results_by_input.fit_results{m_fit, 3, cnt_samp, m_sim, 3}.fval+...
            %                 2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        end
    end
end
BIC_temp_by_input = BIC_temp_by_input([1 2 3 4 6],:,:);
BIC_temp_by_input = BIC_temp_by_input(:,:,[1 2 3 4 6]);

true_pars = [];
fit_pars = [];
% for cnt_samp=1:100
%     true_pars = [true_pars; parameter_recovery_results.sim_results{6, 3, cnt_samp}.params];
%     fit_pars = [fit_pars; parameter_recovery_results.fit_results{6, 3, cnt_samp, 6, 3}.params];
% end

for cnt_samp=1:100
    true_pars = [true_pars; recovery_results_by_input.sim_results{aaa, 3, cnt_samp}.params];
    fit_pars = [fit_pars; recovery_results_by_input.fit_results{aaa, 3, cnt_samp, aaa, 3}.params];
end

for cnt_samp=1:100
    true_pars = [true_pars; recovery_results_by_attn.sim_results{aaa, 3, cnt_samp}.params];
    fit_pars = [fit_pars; recovery_results_by_attn.fit_results{aaa, 3, cnt_samp, aaa, 3}.params];
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
caxis([0.0 1.0]);
xticks(1:5)
xticklabels(all_model_names_legend)
xlabel('Fit model', 'FontSize', 20)
yticks(1:5)
yticklabels(all_model_names_legend)
ylabel('Simulated model', 'FontSize', 20)
% colorbar()
[txs, tys] = meshgrid(1:5, 1:5);
colormap(flipud(bone))
txts = text(txs(:)-0.2, tys(:), string(num2str(conf_mat_pxp(:), '%.2f')), 'FontSize', 16);
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
xlabel('Fit model', 'FontSize', 20)
yticks(1:10)
yticklabels(xlabels)
ylabel('Simulated model', 'FontSize', 20)
caxis([0. 1])
% colorbar()
% title('Confusion Matrix')
[txs, tys] = meshgrid(1:10, 1:10);
colormap(flipud(bone))
txts = text(txs(:)-0.3, tys(:), string(num2str(conf_mat_pxp(:), '%.2f')), 'FontSize', 12);
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
caxis([-0.9, 0.9])
% colorbar
param_names = {'bias', '\beta', '\omega', 'd', '\alpha_+', '\alpha_-', '\gamma'};
xticklabels(param_names);
yticklabels(param_names);
colormap bluewhitered
xlabel('True parameters', 'FontSize', 20)
ylabel('Fit parameters', 'FontSize', 20)
[txs, tys] = meshgrid(1:7, 1:7);
[param_corr_mat, ps] = corr(true_pars, fit_pars, 'type', 'spearman');
txts = text(txs(:)-0.4, tys(:), string(num2str(param_corr_mat(:), '%.2f')), 'FontSize', 18);
for i=1:7
    for j=1:7
        if (param_corr_mat(i,j)>0.40)
            txts((j-1)*7+i).Color = [1 1 1];
        end
    end
end

figure
for i=1:7
    subplot(3,3,i)
    if (i==2 || i==7)
        scatter(log10(true_pars(fit_pars(:,i)>1e-4,i)), log10(fit_pars(fit_pars(:,i)>1e-4,i)),[],[0.4, 0.4, 0.4]); hold on
        xlim(max([xlim; ylim]))
        ylim(max([xlim; ylim]))
        plot(xlim, ylim, 'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
        ll = lsline;
        xticks(0:3)
        yticks(0:3)
        xticklabels("10^{"+xticks+"}");
        yticklabels("10^{"+yticks+"}");
        xlim('tight')
        ylim('tight')
    else
        scatter(true_pars(:,i), fit_pars(:,i),[],[0.4, 0.4, 0.4]); hold on
        xlim(max([xlim; ylim]))
        ylim(max([xlim; ylim]))
        plot(xlim, ylim, 'Color', [0.5 0.5 0.5]', 'LineWidth', 1, 'LineStyle', '--')
        ll = lsline;
        xlim('tight')
        ylim('tight')
    end
    %     pbaspect([1 1 1])
    set(ll, 'linewidth', 1, 'color', 'black');
    title(param_names{i})
end