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

set(0,'defaultAxesFontSize',25)
%% load subject files
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


% make names for plotting
all_model_names_legend = ["F+C_{joint}", "Exemplar_{static}", "Exemplar_{dynamic}"];


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
% idxperf = ~idxperf;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);


%% load model fits
attns = load('../files/RPL2Analysis_Attention_merged_rep40_500_log.mat');
exemplar = load('../files/RPL2Analysis_Exemplar.mat');

for cnt_sbj = 1:length(idxperf)
    num_params = length(attns.fit_results{5, 3, idxperf(cnt_sbj)}.params);
    lls(1, cnt_sbj) = attns.fit_results{5, 3, idxperf(cnt_sbj)}.fval;
    AICs(1, cnt_sbj) = 2*lls(1, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
    BICs(1, cnt_sbj) = 2*lls(1, cnt_sbj)+log(ntrials)*num_params;
    Rsqs(1, cnt_sbj) = 1-BICs(1, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);

    for m = 1:2
        num_params = length(exemplar.fit_results{m, idxperf(cnt_sbj)}.params);
        lls(m+1, cnt_sbj) = exemplar.fit_results{m, idxperf(cnt_sbj)}.fval;
        AICs(m+1, cnt_sbj) = 2*lls(m+1, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        BICs(m+1, cnt_sbj) = 2*lls(m+1, cnt_sbj)+log(ntrials)*num_params;
        Rsqs(m+1, cnt_sbj) = 1-BICs(m+1, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);
    end
end

%% BMS

[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(-BICs(1:3,:)'/2, ...
    mat2cell((1:3)', repmat([1], 1, 3)));
disp(bor_BIC);


%% plot baseline

imagesc(alpha_BIC'/sum(alpha_BIC));
txts = text((1:3)-0.2, ones(1, 3), string(num2str(pxp_BIC(:), '%.2f')), 'FontSize',20);
for i=1:3
    if (alpha_BIC(i)/sum(alpha_BIC)>0.3)
        txts(i).Color = [1 1 1];
    end
end

xticks(1:10)
xticklabels(all_model_names_legend)
yticks([])

axis image;

h=gca;
h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];

caxis([0 1])
colormap(flipud(bone))
cb = colorbar('southoutside');


cb.Label.String = 'Posterior model probability';
cb.Label.FontSize = 20;


%%

attn_results = [exemplar.fit_results{1, :}];
% attn_results_no_attn = [attns.fit_results{5, 1, :}];
curr_params = reshape([attn_results.params], 8, [])';
% curr_params_no_attn = reshape([attn_results_no_attn.params], 6, [])';
param_names = ["bias", "\beta", "d", "\alpha_+", "\alpha_-", "w_1", "w_2", "w_3"];
% figure
% curr_params(:,[2 7]) = log(curr_params(:,[2 7])+1e-4);
% curr_params(:,[3 4 5 6]) = log(curr_params(:,[3 4 5 6])+1e-4) ...
%                          - log(1-curr_params(:,[3 4 5 6])+1e-4);
[S,AX,BigAx,H,HAx] = plotmatrix(curr_params(idxperf,:));
for i=1:8
    H(i).NumBins=10;
end
hold on;
for i=1:8
    xlabel(AX(8,i), param_names{i}, 'FontSize', 14);
    ylabel(AX(i,1), param_names{i}, 'FontSize', 14);
end

%%

histogram(Rsqs(2,:), 'BinEdges', -0.05:0.05:0.7, 'Normalization', 'probability');hold on
histogram(Rsqs(3,:), 'BinEdges', -0.05:0.05:0.7, 'Normalization', 'probability');
xline(0, '--', 'LineWidth', 1);
legend('Static', 'Dynamic', 'Chance', 'Location', 'northeast');
xlabel("Pseudo-R^2");
ylabel('Prop. Subject');
box off

%%
normed_static_ws = curr_params(:,6:end)./sum(curr_params(:,6:end),2);

b = bar(mean(normed_static_ws(:,[2,1,3]),1)); hold on
errorbar(1:3, mean(normed_static_ws(:,[2,1,3]),1),...
    std(normed_static_ws(:,[2,1,3]),1,1)/sqrt(size(normed_static_ws,1)),...
    "ko", "LineWidth",1);

xticklabels(["Inf", "Noninf1", "Noninf2"])
xlabel('Feature dimension');
ylabel('Normalized attn. weights')
xlim([0.4, 3.6]);

b.FaceColor = 'flat';
b.CData = colormap('lines(3)');

