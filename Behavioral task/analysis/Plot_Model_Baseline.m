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

set(0,'defaultAxesFontSize',16)
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
all_model_names_legend = ["F+C_{joint}", "F+C_{inf}", ...
                            "F+C_{noninf1}", "F+C_{noninf2}"];


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
static_conj = load('../files/RPL2Analysis_Baseline_ConjunctionBased.mat');
static_obj = load('../files/RPL2Analysis_Baseline_ObjectBased.mat');


for cnt_sbj = 1:length(idxperf)
    num_params = length(attns.fit_results{5, 3, idxperf(cnt_sbj)}.params);
    lls(1, cnt_sbj) = attns.fit_results{5, 3, idxperf(cnt_sbj)}.fval;
    AICs(1, cnt_sbj) = 2*lls(1, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
    BICs(1, cnt_sbj) = 2*lls(1, cnt_sbj)+log(ntrials)*num_params;
    Rsqs(1, cnt_sbj) = 1-BICs(1, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);

    for m = 1:3
        num_params = length(static_conj.fit_results{m, idxperf(cnt_sbj)}.params);
        lls(m+1, cnt_sbj) = static_conj.fit_results{m, idxperf(cnt_sbj)}.fval;
        AICs(m+1, cnt_sbj) = 2*lls(m+1, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
        BICs(m+1, cnt_sbj) = 2*lls(m+1, cnt_sbj)+log(ntrials)*num_params;
        Rsqs(m+1, cnt_sbj) = 1-BICs(m+1, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);
    end

    num_params = length(static_obj.fit_results{1, idxperf(cnt_sbj)}.params);
    lls(5, cnt_sbj) = static_obj.fit_results{1, idxperf(cnt_sbj)}.fval;
    AICs(5, cnt_sbj) = 2*lls(5, cnt_sbj)+2*num_params+(2*num_params*(num_params-1))/(ntrials-num_params-1);
    BICs(5, cnt_sbj) = 2*lls(5, cnt_sbj)+log(ntrials)*num_params;
    Rsqs(5, cnt_sbj) = 1-BICs(5, cnt_sbj)./(-2*logsigmoid(0)*ntrials+log(ntrials)*num_params);
end

%% BMS

[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(-BICs(1:4,:)'/2, ...
    mat2cell((1:4)', repmat([1], 1, 4)));
disp(bor_BIC);


%% plot baseline

imagesc(alpha_BIC'/sum(alpha_BIC));
txts = text((1:4)-0.2, ones(1, 4), string(num2str(pxp_BIC(:), '%.2f')), 'FontSize',20);
for i=1:4
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

%% compare with obj

[alpha_obj,exp_r_obj,xp_obj,pxp_obj,bor_obj,g_obj] = bms(-BICs([1,5],:)'/2, ...
    mat2cell((1:2)', repmat([1], 1, 2)));
disp(bor_obj);

imagesc(alpha_obj'/sum(alpha_obj));
txts = text((1:2)-0.1, ones(1, 2), string(num2str(pxp_obj(:), '%.2f')), 'FontSize',20);
for i=1:2
    if (alpha_obj(i)/sum(alpha_obj)>0.3)
        txts(i).Color = [1 1 1];
    end
end

xticks(1:2)
xticklabels({'F+C_{joint}', 'O'})
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

