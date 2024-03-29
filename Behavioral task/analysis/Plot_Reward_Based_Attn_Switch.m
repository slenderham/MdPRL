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

set(0,'defaultAxesFontSize',22)
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
    "fMLchoiceLL_RL2conjdecayattn_spread", ...
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
    flaginfs(cnt_sbj) = expr.flaginf;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
flaginfs = flaginfs(idxperf);
% idxperf = 1:length(subjects);

%%
all_Xs_for_lr = [];
all_Ys_for_lr = [];
lags_to_fit = 2;
trials_to_fit = lags_to_fit:150;

for cnt_sbj = 1:length(idxperf)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{idxperf(cnt_sbj)} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{idxperf(cnt_sbj)} , '.mat'] ;

    inputs_struct = load(inputname);
    results_struct = load(resultsname);

    inputTarget   = inputs_struct.input.inputTarget;
    Ntrials      = length(results_struct.results.reward);
    rewards      = results_struct.results.reward;
    choices      = results_struct.results.choice;
    % reward
    rew = (rewards(trials_to_fit)==1);
    rew = reshape(repmat(rew, 1, 6), [], 1);
    unr = (rewards(trials_to_fit)==0);
    unr = reshape(repmat(unr, 1, 6), [], 1);

    flat_attn_x = reshape(squeeze(attn_ws(cnt_sbj, trials_to_fit, [1 2 3 1 2 3])), [], 1);
    flat_attn_y = reshape(squeeze(attn_ws(cnt_sbj, trials_to_fit+1, [1 2 3 1 2 3])), [], 1);
    
%     [~, dim_max] = max(squeeze(value_pdists(cnt_sbj, trials_to_fit, [1 2 3])), [], 2);

    flat_vd_x = reshape(squeeze(value_pdists(cnt_sbj, trials_to_fit, [1 2 3 5 4 6])), [], 1);
    flat_vd_y = reshape(squeeze(value_pdists(cnt_sbj, trials_to_fit+1, [1 2 3 5 4 6])), [], 1);

    Xs_for_lr = [rew-unr, flat_vd_x, (rew-unr).*flat_attn_x];

    Ys_for_lr = zscore(flat_vd_y-flat_vd_x);
    all_Xs_for_lr = [all_Xs_for_lr; zscore(Xs_for_lr,[],1) ones(size(Xs_for_lr, 1),1)*cnt_sbj];
    all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr];
end

tbl_to_fit = array2table( ...
    [all_Ys_for_lr, all_Xs_for_lr], ...
    "VariableNames", ["vd_next", "r", "vd_past", "rXa", "subject"]);
mdl = fitglme(tbl_to_fit, "vd_next~r+vd_past+rXa+(r+vd_past+rXa|subject)",...
     "CovariancePattern", "Diagonal", "Verbose", 0)

%%
bar([mean(Ys_for_lr(Xs_for_lr(:,1)==1)); mean(Ys_for_lr(Xs_for_lr(:,1)==-1))], 'FaceColor',[.5 .5 .5]);hold on;
errorbar([mean(Ys_for_lr(Xs_for_lr(:,1)==1)); mean(Ys_for_lr(Xs_for_lr(:,1)==-1))],...
[std(Ys_for_lr(Xs_for_lr(:,1)==1))/sqrt(numel(Ys_for_lr(Xs_for_lr(:,1)==1))); std(Ys_for_lr(Xs_for_lr(:,1)==-1))/sqrt(numel(Ys_for_lr(Xs_for_lr(:,1)==1)))],...
'k', "LineStyle","none","Linewidth",1)
ylim([0.46, 0.74])
xlim([0.5, 2.5])
xticklabels(["Rewarded", "Unrewarded"])
yticks(0:0.05:1.0)
ylabel("P(Switch)")