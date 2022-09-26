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


%% Simulate model with best param
load('../files/simulated_vars')

unfilt_AICs = trial_AICs;
unfilt_BICs = trial_BICs;
unfilt_lls = trial_lls;

trial_AICs=trial_AICs(:,:,idxperf,:);
trial_BICs=trial_BICs(:,:,idxperf,:);
trial_lls=trial_lls(:,:,idxperf,:);

%% Calculate the value differentials along different dimensions

all_diff_vals = [];
all_choicetrials = [];


clear diff_vals
for cnt_sbj = 1:length(idxperf)
%     disp(strcat("Fitting subject ", num2str(cnt_sbj)));
    inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(idxperf(cnt_sbj)) , ".mat") ;
    resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(idxperf(cnt_sbj)) , ".mat") ;

    inputs_struct = load(inputname);
    results_struct = load(resultsname);

    expr = results_struct.expr;
    input = inputs_struct.input;
    results = results_struct.results;

    shapeMap = repmat([1 2 3 ;
        1 2 3 ;
        1 2 3 ], 1,1,3) ;

    colorMap = repmat([1 1 1 ;
        2 2 2 ;
        3 3 3], 1,1,3) ;

    patternMap(:,:,1) = ones(3,3) ;
    patternMap(:,:,2) = 2*ones(3,3) ;
    patternMap(:,:,3) = 3*ones(3,3) ;

    choicetrials = results.choice ;
    inputTarget = input.inputTarget ;

    curr_subj_vals = all_values{5, 3, idxperf(cnt_sbj)};
    curr_subj_vals = [0.5*ones(36,1) curr_subj_vals];

    for cnt_trial=1:ntrials
        vf = curr_subj_vals(1:9,cnt_trial);
        vc = curr_subj_vals(10:36,cnt_trial);
        
        idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ; % 1-3
        idx_color(2)    = colorMap(inputTarget(2, cnt_trial))+3 ; % 4-6
        idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial))+6 ; % 7-9
        idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
        idx_color(1)    = colorMap(inputTarget(1, cnt_trial))+3 ;
        idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial))+6 ;
        idx_patternshape(1) = (idx_pattern(1)-7)*3 + idx_shape(1) ; % 1-9
        idx_patternshape(2) = (idx_pattern(2)-7)*3 + idx_shape(2) ; 
        assert(1<=idx_patternshape(1) & idx_patternshape(1)<=9 & 1<=idx_patternshape(2) & idx_patternshape(2)<=9);
        idx_patterncolor(1) = (idx_pattern(1)-7)*3 + (idx_color(1)-4)+10 ; % 10-18
        idx_patterncolor(2) = (idx_pattern(2)-7)*3 + (idx_color(2)-4)+10 ;
        assert(10<=idx_patterncolor(1) & idx_patterncolor(1)<=18 & 10<=idx_patterncolor(2) & idx_patterncolor(2)<=18);
        idx_shapecolor(1) = (idx_shape(1)-1)*3 + (idx_color(1)-4)+19 ;
        idx_shapecolor(2) = (idx_shape(2)-1)*3 + (idx_color(2)-4)+19 ; % 19-27
        assert(19<=idx_shapecolor(1) & idx_shapecolor(1)<=27 & 19<=idx_shapecolor(2) & idx_shapecolor(2)<=27);

        diff_vals(cnt_trial,:) = [vf(idx_shape(2))-vf(idx_shape(1)) ...
                                  vf(idx_color(2))-vf(idx_color(1)) ...
                                  vf(idx_pattern(2))-vf(idx_pattern(1)) ...
                                  vc(idx_patterncolor(2))-vc(idx_patterncolor(1)) ...
                                  vc(idx_patternshape(2))-vc(idx_patternshape(1)) ...
                                  vc(idx_shapecolor(2))-vc(idx_shapecolor(1))];
    end
%     diff_vals = (diff_vals-mean(diff_vals, 1))./(std(diff_vals, 1));
    all_diff_vals = [all_diff_vals; diff_vals ones(cnt_trial,1)*cnt_sbj];
    all_choicetrials = [all_choicetrials; choicetrials-1];
end

tbl = array2table([all_choicetrials all_diff_vals], ...
    'VariableNames', ["Choice", "S", "C", "P", "PC", "PS", "SC", "subject"]);
mdl = fitglme(tbl, "Choice ~ S+C+P+PC+PS+SC+(S+C+P+PC+PS+SC|subject)", 'Distribution','binomial');

figure
errorbar(mdl.Coefficients.Estimate([3 2 4 6 5 7]), mdl.Coefficients.SE([3 2 4 6 5 7]), 'ko');
ylim([0, 10])
xlim([0, 7])
xticks(1:6)
xticklabels(["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf2}"])
ylabel("Regression Weights")