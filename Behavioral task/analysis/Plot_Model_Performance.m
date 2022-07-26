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

attns = load('../files/RPL2Analysis_Attention_merged_rep50.mat') ;

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;            
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
        end
    end
end

%% bayesian model selection

[alpha_AIC,exp_r_AIC,xp_AIC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_AIC);
% figure;
% h = bar(reshape(pxp_AIC, 10, 5));hold on;
% 
% hx_pos = nan(10, 5);
% for i = 1:5
%     hx_pos(:,i) = h(i).XEndPoints;
% end
% 
% plot(hx_pos(:), alpha_AIC/sum(alpha_AIC), 'ko', 'DisplayName', "E[Freq]")
% xticks(1:10)
% xticklabels(attn_modes_legend)
% set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
% ylabel('pxp')
% legend()


[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_BIC);
% figure;
% h = bar(reshape(pxp_BIC, 10, 5));hold on;
% 
% hx_pos = nan(10, 5);
% for i = 1:5
%     hx_pos(:,i) = h(i).XEndPoints;
% end
% 
% plot(hx_pos(:), alpha_BIC/sum(alpha_BIC), 'ko', 'DisplayName', "E[Freq]")
% xticks(1:10)
% xticklabels(attn_modes_legend)
% set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
% ylabel('pxp')
% legend()

[~, best_model_inds] = max(g_BIC);
% title('Posterior Model Probability')

[alpha_input,exp_r_input,xp_input,pxp_input,bor_input,g_input] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [10, 5])', repmat([1], 1, 5)));
disp(bor_input);
% figure;
% h = bar(pxp_input);hold on;
% plot(alpha_input/sum(alpha_input), 'ko', 'DisplayName', "E[Freq]")
% xticks(1:5)
% xticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'})
% ylabel('pxp')

[alpha_attn,exp_r_attn,xp_attn,pxp_attn,bor_attn,g_attn] = bms(reshape(-BICs/2, [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [5, 10])', repmat([1], 1, 10)));
disp(bor_attn);
% figure;
% h = bar(pxp_attn);hold on;
% plot(alpha_attn/sum(alpha_attn), 'ko', 'DisplayName', "E[Freq]")
% xticks(1:10)
% xticklabels(attn_modes_legend(:,1))
% ylabel('pxp')

t = tiledlayout(5, 7, 'TileSpacing','compact');
nexttile([1 6])
% imagesc(alpha_attn'/sum(alpha_attn));
imagesc(pxp_attn)
caxis([0 1])
xticks([])
yticks([])
nexttile
axis off
nexttile([4 6])
imagesc(reshape(alpha_BIC/sum(alpha_BIC), 10, 5)');
imagesc(reshape(pxp_BIC, 10, 5)');
caxis([0 1])
xticks(1:10)
xticklabels(attn_modes_legend)
yticks(1:5)
yticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
nexttile([4 1])
% imagesc(alpha_input/sum(alpha_input));
imagesc(pxp_input')
caxis([0 1])
xticks([])
yticks([])
colormap(flipud(bone))
cb = colorbar;
cb.Layout.Tile = 'South';


%% Plot All Parameters
attn_results = [attns.fit_results{5, 3, :}];
curr_params = reshape([attn_results.params], 6, [])';
param_names = {'bias', '\beta', '\omega', 'decay', '\alpha_+', '\alpha_-', '\gamma'};
[S,AX,BigAx,H,HAx] = plotmatrix(curr_params(idxperf, :));
for i=1:6
H(i).NumBins=20;
end
hold on;
for i=1:6
xlabel(AX(6,i), param_names{i}, 'FontSize', 18);
ylabel(AX(i,1), param_names{i}, 'FontSize', 18);
end

%% Focus on gamma and omega
figure;
subplot(121);
hf = histfit(max(curr_params(idxperf, 7), 0), 10, 'kernel');hf(1).FaceColor=rgb('grey');hf(2).Color = [0 0 0]';
xlabel('\gamma', 'FontSize', 25)
xlim([-50, 1050]);xticks(0:200:1000);
ylabel('Density')
subplot(122);
hf = histfit(max(min(curr_params(idxperf, 3), 1-1e-4), 1e-4), 10, 'kernel');hf(1).FaceColor=rgb('grey');hf(2).Color = [0 0 0]';
xlim([-0.05, 1.05]);xticks(0:0.2:1);
xlabel('\omega', 'FontSize', 25)

%% learning rate bias
violinplot(curr_params(idxperf, 5:6), [], 'Width', 0.3, 'Bandwidth', 0.1)
ylim([0, 1.05]);
xticklabels(["\alpha_+", "\alpha_-"]);
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',25);