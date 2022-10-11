clc
clear
close all
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../files")
addpath("../models")
addpath("../utils")
% addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
% addpath("../utils/vbmc")

set(0,'defaultAxesFontSize',18)
%% load result files
% feat = load('../files/RPL2Analysisv3_5_FeatureBased') ;
% obj = load('../files/RPL2Analysisv3_5_FeatureObjectBased') ;
% conj  = load('../files/RPL2Analysisv3_5_ConjunctionBased') ;

subjects1 = [...
    "AA", "AB", "AC", "AD", "AE", "AF", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY", "AZ", "BA", "BB", "BC", "BD", ...
    "BE", "BF", "BG", "BH", "BI", "BJ", "BK", "BL", ...
    "BM", "BN", "BO", "BP", "BQ", "BR", "CC", "DD", ...
    "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", ...
    "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", ...
    "UU", "VV", "WW", "XX", "YY", "ZZ"];
subjects1 = lower(subjects1);
subjects1_inputs = "inputs/input_"+subjects1;
subjects1_prl = "SubjectData/PRL_"+subjects1;

subjects2 = [...
    "AA", "AB", "AC", "AD", "AE", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY"] ;
subjects2_inputs = "inputs2/input_"+subjects2;
subjects2_prl = "SubjectData2/PRL_"+subjects2;

subjects_inputs = [subjects1_inputs subjects2_inputs];
subjects_prl = [subjects1_prl subjects2_prl];

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
    "fMLchoiceLL_RL2ftobjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn_onlyfattn", ...
    "fMLchoiceLL_RL2conjdecayattn_constrained"];

% make names for plotting
all_model_names_legend = ["F", "F+O", "F+C_{untied}", "F+C_{feat attn}", "F+C_{tied}"];
attn_modes_legend = strcat(attn_modes(:,1),"X",attn_modes(:,2));
[all_model_names_legend, attn_modes_legend] = meshgrid(all_model_names_legend, attn_modes_legend);
all_legends = strcat(all_model_names_legend(:), "X", attn_modes_legend(:));


ntrials = 432;
ntrialPerf       = 33:432;
% perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH           = 0.53;

cmap = lines(256);

for cnt_sbj = 1:length(subjects_inputs)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

    load(inputname)
    load(resultsname)

    rew(cnt_sbj,:)                = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew(cnt_sbj,:)          = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew(cnt_sbj,ntrialPerf)) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);

% figure
% plot_shaded_errorbar(mean(movmean(rew(idxperf,:), 20, 2))', std(movmean(rew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, 'k');
% plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, [0.5 0.5 0.5]);
% xlabel('Trial Number')
% ylabel('Performance')
% legend({'', 'Reward', '', 'Proportion Better'})
% xlim([0, 432])


%% load results with attn and ML params

attns = load('../files/RPL2Analysis_Attention_merged_rep40_250.mat') ;

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            num_params = length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*num_params;
        end
    end
end

disp(mean(BICs, 3))

%% bayesian model selection

% [alpha_AIC,exp_r_AIC,xp_AIC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs/2, [2 1 3]), [50, length(idxperf)])', ...
%     mat2cell((1:50)', repmat([1], 1, 50)));
% disp(bor_AIC);

[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
    mat2cell((1:50)', repmat([1], 1, 50)));
% disp(bor_BIC);
[~, best_model_inds] = max(g_BIC);


[alpha_input,exp_r_input,xp_input,pxp_input,bor_input,g_input] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
    mat2cell(reshape(1:50, [10, 5])', repmat([1], 1, 5)));
% disp(bor_input);


[alpha_attn,exp_r_attn,xp_attn,pxp_attn,bor_attn,g_attn] = bms(reshape(-BICs/2, [50, length(idxperf)])', ...
    mat2cell(reshape(1:50, [5, 10])', repmat([1], 1, 10)));
% disp(bor_attn);

%% 
t = tiledlayout(5, 7, 'TileSpacing','compact');
nexttile([1 6])
imagesc(alpha_attn'/sum(alpha_attn));
txts = text((1:10)-0.3, ones(1, 10), string(num2str(pxp_attn(:), '%.2f')), 'FontSize',12);
for i=1:10
    if (alpha_attn(i)/sum(alpha_attn)>0.3)
        txts(i).Color = [1 1 1];
    end
end
% imagesc(pxp_attn)
caxis([0 1])
xticks([])
yticks([])
nexttile
axis off
nexttile([4 6])
imagesc(reshape(alpha_BIC/sum(alpha_BIC), 10, 5)');
[txs, tys] = meshgrid(1:10, 1:5);
txs = txs';
tys = tys';
txts = text(txs(:)-0.3, tys(:), string(num2str(pxp_BIC(:), '%.2f')),'FontSize',12);
for i=1:5
    for j=1:10
        if (alpha_BIC((i-1)*10+j)/sum(alpha_BIC)>0.3)
            txts((i-1)*10+j).Color = [1 1 1];
        end
    end
end
% imagesc(reshape(pxp_BIC, 10, 5)');
caxis([0 1])
xticks(1:10)
xticklabels(attn_modes_legend)
h=gca;
h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];
xlabel('Attentional mechanisms')
yticks(1:5)
yticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
xtickangle(30)
ylabel('Learning strategies')
nexttile([4 1])
imagesc(alpha_input/sum(alpha_input));
% imagesc(pxp_input')
txts = text(ones(1, 5)-0.3, 1:5, string(num2str(pxp_input(:), '%.2f')), 'FontSize',12);
for i=1:5
    if (alpha_input(i)/sum(alpha_input)>0.3)
        txts(i).Color = [1 1 1];
    end
end
caxis([0 1])
xticks([])
yticks([])
colormap(flipud(bone))
cb = colorbar;
cb.Layout.Tile = 'South';

%% Plot All Parameters
set(0,'defaultAxesFontSize',14)
attn_results = [attns.fit_results{5, 3, idxperf}];
% attn_results_no_attn = [attns.fit_results{5, 1, :}];
curr_params = reshape([attn_results.params], 7, [])';
% curr_params_no_attn = reshape([attn_results_no_attn.params], 6, [])';
param_names = {'bias', '\beta', '\omega', 'd', '\alpha_+', '\alpha_-', '\gamma'};
figure
[S,AX,BigAx,H,HAx] = plotmatrix(curr_params);
for i=1:7
    H(i).NumBins=10;
end
hold on;
for i=1:7
    xlabel(AX(7,i), param_names{i}, 'FontSize', 14);
    ylabel(AX(i,1), param_names{i}, 'FontSize', 14);
end
set(0,'defaultAxesFontSize',18)

%% Focus on gamma and omega
figure;
temp_bound = 250;
% psuedolog = @(x) asinh(x/2)/log(exp(1));
hf = histogram((min(max(curr_params(idxperf, 7), 1e-4), temp_bound-1e-4)), 'NumBins', 10, 'Normalization','pdf'); hold on;
hf(1).FaceColor=rgb('grey');
[f,xi] = ksdensity((min(max(curr_params(idxperf, 7), 1e-4), temp_bound-1e-4)), 'Support', [0, temp_bound], 'BoundaryCorrection', 'Reflection');
plot(xi(2:end-1), f(2:end-1), 'k', 'linewidth', 2)
xlabel('\gamma', 'FontSize', 20)
xlim([0, temp_bound]);xticks(0:50:temp_bound);
ylabel('Density')

figure
% inv_logit = @(x) log(x+1e-3)-log(1-x+1e-3);
hf = histogram((max(min(curr_params(idxperf, 3), 1-eps), eps)), 'NumBins', 10, 'Normalization','pdf'); hold on;
hf(1).FaceColor=rgb('grey');
[f,xi] = ksdensity((max(min(curr_params(idxperf, 3), 1-eps), eps)), 'Support', [0, 1], 'BoundaryCorrection', 'Reflection');
plot(xi(2:end-1), f(2:end-1), 'k', 'linewidth', 2)
xlim([0, 1]);xticks(0:0.2:1);
xlabel('\leftarrow Conjunction                  \omega                  Feature \rightarrow', 'FontSize', 20)
ylabel('Density')

%% learning rate bias
figure
clrmats = [[0.4660 0.6740 0.1880]; [0.6350 0.0780 0.1840]; rgb('grey')];
violinplot(curr_params(idxperf, [5 6 4]), [], 'Width', 0.25, 'ViolinColor', clrmats);
ylim([0, 1.05]);
xlim([0.5, 3.5]);
xticklabels(["\alpha_+", "\alpha_-", "d"]);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',25);
% title('DiffXL', 'fontsize', 20)

% figure
% clrmats = [[0.4660 0.6740 0.1880]; [0.6350 0.0780 0.1840]; rgb('grey')];
% violinplot(curr_params_no_attn(idxperf, [5 6 4]), [], 'Width', 0.25, 'ViolinColor', clrmats);
% ylim([0, 1.05]);
% xlim([0.5, 3.5]);
% xticklabels(["\alpha_+", "\alpha_-", "d"]);
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',25);
% % title('No Attn', 'fontsize', 20)

%%
figure
pbaspect([2 1 1])
inds = [5,6,4];
t = tiledlayout(1,3,'TileSpacing','compact');
for i=1:3
    nexttile
    scatter(curr_params(idxperf, inds(i)), curr_params_no_attn(idxperf, inds(i)), ...
        40, clrmats(i,:));hold on;
    max_lim = max(max(curr_params(idxperf, inds(i)), curr_params_no_attn(idxperf, inds(i))));
    xlim([0, max_lim]);
    ylim([0, max_lim]);
    plot([0,1],[0,1],'--k')
    xticks(0:floor(max_lim*2.5)*0.2:1);
    yticks(0:floor(max_lim*2.5)*0.2:1);
    title(param_names{inds(i)})
end
xlabel(t, 'diffXL', 'FontSize', 18)
ylabel(t, 'no attn', 'FontSize', 18)

figure
t = tiledlayout(1,3,'TileSpacing','compact');
for i=1:3
    nexttile
    histogram(curr_params(idxperf, inds(i))-curr_params_no_attn(idxperf, inds(i)), ...
        'FaceColor', clrmats(i,:), 'BinEdges', -1:0.2:1, 'Normalization', 'probability');
    ylim([0, 1])
    if i>1
        yticklabels([])
    end
    xlabel("\Delta"+string(param_names{inds(i)}))
end

ylabel(t, 'Frequency', 'FontSize', 18)