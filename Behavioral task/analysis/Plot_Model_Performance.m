clc
clear
close all
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../files")
addpath("../models")
addpath("../utils")
addpath("../utils/DERIVESTsuite/DERIVESTsuite/")
addpath("../utils/vbmc")
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
    
    rew{cnt_sbj}                  = results.reward ;
    [~, idxMax]                   = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew{cnt_sbj}            = results.choice' == idxMax ;
    perfMean(cnt_sbj)             = nanmean(choiceRew{cnt_sbj}(ntrialPerf)) ;
end
idxperf = find(perfMean>=perfTH);
% idxperf = 1:length(subjects);

%% load results with attn and ML params

attns = load('../files/RPL2Analysis_Attention.mat') ;
for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;            
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
        end
    end
end

%% Simulate model with best param
for m = 1:length(all_model_names)
    disp("=======================================================");
    disp(strcat("Fitting model ", all_model_names(m)));
    basic_params = cell(length(subjects_inputs)); % store the attention-less model's parameters for each model type
    for a = 1:length(attn_modes)
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        for cnt_sbj = 1:length(subjects_inputs)
%             disp(strcat("Fitting subject ", num2str(cnt_sbj)));
            inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(cnt_sbj) , ".mat") ;
            resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;

            inputs_struct = load(inputname);
            results_struct = load(resultsname);
% 
            expr = results_struct.expr;
            input = inputs_struct.input;
            results = results_struct.results;

            expr.shapeMap = repmat([1 2 3 ;
                1 2 3 ;
                1 2 3 ], 1,1,3) ;

            expr.colorMap = repmat([1 1 1 ;
                2 2 2 ;
                3 3 3], 1,1,3) ;

            expr.patternMap(:,:,1) = ones(3,3) ;
            expr.patternMap(:,:,2) = 2*ones(3,3) ;
            expr.patternMap(:,:,3) = 3*ones(3,3) ;
            
            sesdata = struct();
            sesdata.input   = input ;
            sesdata.expr    = expr ;
            sesdata.results = results ;
            sesdata.NtrialsShort = expr.NtrialsShort ;
            sesdata.flagUnr = 1 ;

            sesdata.flag_couple = 0 ;
            sesdata.flag_updatesim = 0 ;


            % load attn type (const, diff, sum, max) and attn
            % time(none, choice, learning, both)
            sesdata.attn_op = attn_modes(a,1);
            sesdata.attn_time = attn_modes(a,2);

            % load best params
            best_pars = attns.fit_results{m, a, cnt_sbj}.params;


            % load model likelihood func and optimize
            ll = str2func(all_model_names(m));

            trial_lls(m, a, cnt_sbj, :) = ll(best_pars, sesdata);
            trial_AICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+2*length(best_pars)/ntrials;
            trial_BICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+log(ntrials)*length(best_pars)/ntrials;
        end
    end
end

%% bayesian model selection

[alpha_AIC,exp_r_AIC,xp_BAC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs, [2 1 3])/2, [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_AIC);
figure;
h = bar(reshape(pxp_AIC, 10, 5));
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()



[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs, [2 1 3])/2, [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_BIC);
figure;
h = bar(reshape(pxp_BIC, 10, 5));
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()
% title('Posterior Model Probability')

%% Plot Model Evidence

wSize = 50;
clrmat = colormap('lines(5)') ;

smth_AIC = movmedian(trial_AICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_BIC = movmedian(trial_BICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_ll = movmedian(trial_lls, [0 wSize-1], 4, 'Endpoints', 'discard');

for i=1:5
    [~, min_attn_type] = min(mean(BICs(i,:,:), 3));
    l(i) = plot_shaded_errorbar(squeeze(mean(smth_BIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(std(smth_BIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(10*length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])

xlabel('Trial')
ylabel('Trial-wise BIC')

figure
for i=1:5
    [~, min_attn_type] = min(mean(AICs(i,:,:), 3));
    l(i) = plot_shaded_errorbar(squeeze(mean(smth_AIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(std(smth_AIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(10*length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])

xlabel('Trial')
ylabel('Trial-wise AIC')