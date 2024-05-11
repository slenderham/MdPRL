clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../files")
addpath("../models")
addpath("../utils")

%%
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
% subjects1 = ["AA", "AB"];
subjects1 = lower(subjects1);
subjects1_inputs = "inputs/input_"+subjects1;
subjects1_prl = "SubjectData/PRL_"+subjects1;

subjects2 = [...
    "AA", "AB", "AC", "AD", "AE", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY"] ;
% subjects2 = ["AA", "AB"] ;
subjects2_inputs = "inputs2/input_"+subjects2;
subjects2_prl = "SubjectData2/PRL_"+subjects2;

subjects_inputs = [subjects1_inputs subjects2_inputs];
subjects_prl = [subjects1_prl subjects2_prl];


attns = load('../files/RPL2Analysis_Baseline_ConjunctionBased.mat') ;
% attns = load('../files/RPL2Analysis_Attention_lim_temp_500_6models_40_rpe.mat') ;
ntrials = 432;

ntrialPerf       = 33:432;
% perfTH           = 0.5 + 2*sqrt(.5*.5/length(ntrialPerf)) ;
perfTH           = 0.53;

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

for cntD = 1:3
    disp("=======================================================");
    disp(strcat("Running F+C dimension ", string(cntD)));
    parfor cnt_sbj = 1:length(subjects_inputs)
        disp(['Subject: ', num2str(cnt_sbj)])
        inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(cnt_sbj) , ".mat") ;
        resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;
        
        inputs_struct = load(inputname);
        results_struct = load(resultsname);
    
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
        sesdata.cntD = cntD ;
        sesdata.flag_couple = 0 ;
        sesdata.flag_updatesim = 0 ;
    
        NparamBasic = 4 ;
        if sesdata.flagUnr==1
            sesdata.Nalpha = 2 ;
        else
            sesdata.Nalpha = 1 ;
        end

        % load best params
        best_pars = attns.fit_results{cntD, cnt_sbj}.params;

        [trial_ll, latents] = fMLchoiceLL_RL2conjdecay(best_pars, sesdata);

        all_values{cntD, cnt_sbj} = latents.V;
        trial_lls(cntD, cnt_sbj, :) = trial_ll;
        trial_AICs(cntD, cnt_sbj, :) = 2*trial_lls(cntD, cnt_sbj, :)+2*length(best_pars)/ntrials;
        trial_BICs(cntD, cnt_sbj, :) = 2*trial_lls(cntD, cnt_sbj, :)+log(ntrials)*length(best_pars)/ntrials;
        trial_R2s(cntD, cnt_sbj, :) = 1-trial_BICs(cntD, cnt_sbj, :)./(2*log(2)+log(ntrials)*length(best_pars)/ntrials);
    end
end

%%
trial_BICs = trial_BICs(:,idxperf,:);

%%
wSize=100;
clrmat = colormap('lines(5)');

smth_trial_BICs = movmean(trial_BICs, wSize, 3, 'Endpoint', 'discard');

for d=[1 2 3]
%     plot_shaded_errorbar(squeeze(mean(smth_attn_ws(:,:,d), 1))', ...
%         squeeze(std(smth_attn_ws(:,:,d), [], 1))'/sqrt(length(idxperf)), ...
%         wSize, clrmat(d,:));hold on;

    pse(d) = plot_shaded_errorbar(squeeze(mean(smth_trial_BICs(d,:,:)-mean(smth_trial_BICs, 1))), ...
        squeeze(std(smth_trial_BICs(d,:,:)-mean(smth_trial_BICs, 1)))/sqrt(length(idxperf)), ...
        wSize, clrmat(d,:));hold on;
end

axis tight
% xlim([wSize, ntrials-wSize+1]);
ylim([-0.02, 0.03]);
legend(pse, ["F+C_{inf}", "F+C_{noninf1}", "F+C_{noninf2}"], "Location", "north", "Orientation","horizontal")

xlabel('Trial')
ylabel('Trialwise BIC')
