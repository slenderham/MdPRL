clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath('../utils')

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

ntrialPerf       = 33:432;
perfTH = 0.53;
idxSubject       = 1:length(subjects_inputs);
wSize = 20 ;

for cnt_sbj = 1:length(subjects_inputs)
    if mod(cnt_sbj, 10)==0
        disp(cnt_sbj)
    end
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj}, '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj}, '.mat'] ;

    load(inputname)
    load(resultsname)

    rew{cnt_sbj}        = results.reward ;
    [~, idxMax]         = max(expr.prob{1}(input.inputTarget)) ;
    choiceRew{cnt_sbj}  = results.choice' == idxMax ;
    perfMean(cnt_sbj)   = nanmean(choiceRew{cnt_sbj}(ntrialPerf)) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = false;
% idxperf(36) = false;
subjects_inputs = subjects_inputs(idxperf);
subjects_prl = subjects_prl(idxperf);

%% logistic regression
% C_t ~ beta [C_{t-i}Xrew_{t-1}-C_{t-1}Xunr_{t-1}]
% each term coded as binary variable:
%   1. chosen and rewarded,
%   2. chosen and unrewarded,
% all of them are zero if that instance not present during that trial
% do separately for each feature and each conjunction (each object?)
% two types of generalization: see feature, change conjunction
%                           or see conjunction, change feature

% lags_to_fit = 3; % how far back should the CxR variables be included

% obj to ft/conj mappings
shapeMap = repmat([1 2 3 ;
    1 2 3 ;
    1 2 3 ], 1,1,3) ;

colorMap = repmat([1 1 1 ;
    2 2 2 ;
    3 3 3], 1,1,3) ;

patternMap(:,:,1) = ones(3,3) ;
patternMap(:,:,2) = 2*ones(3,3) ;
patternMap(:,:,3) = 3*ones(3,3) ;

patterncolorMap = 3*(patternMap-1)+colorMap;
patternshapeMap = 3*(patternMap-1)+shapeMap;
shapecolorMap = 3*(shapeMap-1)+colorMap;

lags_to_fit = 1;

clear all_Xs_for_lr all_Ys_for_lr lr_weights_mean

for cnt_O = 1:27
    all_Xs_for_lr = [];
    all_Ys_for_lr = [];
    disp(num2str(cnt_O));
    for cnt_sbj = 1:length(subjects_inputs)
        inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
        resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

        inputs_struct = load(inputname);
        results_struct = load(resultsname);

        inputTarget   = inputs_struct.input.inputTarget;
        Ntrials      = length(results_struct.results.reward);
        rewards      = results_struct.results.reward;
        choices      = results_struct.results.choice;

        % the object chosen and unchosen
        targCh = inputs_struct.input.inputTarget(...
            results_struct.results.choice'+2*(0:(results_struct.expr.Ntrials-1))) ;
        targUnch = inputs_struct.input.inputTarget(...
            (3-results_struct.results.choice)'+2*(0:(results_struct.expr.Ntrials-1))) ;

        % reward
        idx_rew = rewards==1;
        idx_unr = rewards==0;

        % choice
        Ct = targCh==cnt_O;
        uCt = targUnch==cnt_O;

        Xs_for_lr = [];
        Ys_for_lr = [];

        % for each time O_ch is an option
        % look back to find the last time O_fb is an option
        for l = 2:Ntrials
            if (Ct(l) || uCt(l))  % if the object can be chosen at that trial
                Ys_for_lr = [Ys_for_lr; Ct(l)];
                Xs_all_maps = [];
                for i_dim=1:6
                    switch i_dim
                        case 1
                            obj2dim_map = shapeMap;
                        case 2
                            obj2dim_map = colorMap;
                        case 3
                            obj2dim_map = patternMap;
                        case 4
                            obj2dim_map = patterncolorMap;
                        case 5
                            obj2dim_map = patternshapeMap;
                        case 6
                            obj2dim_map = shapecolorMap;
                        otherwise
                    end
                    k = l-1; % start from the previous trial and go back
                    trials_found = 0;
                    Xs_curr_map = [];
                    while k>=1 && trials_found<lags_to_fit
                        if obj2dim_map(cnt_O)==obj2dim_map(targCh(k))
                            Xs_curr_map = [Xs_curr_map idx_rew(k)-idx_unr(k)];
                            trials_found = trials_found + 1;
                        end
                        k = k-1;
                    end
                    Xs_curr_map = [Xs_curr_map zeros(1, lags_to_fit-trials_found)];
                    Xs_all_maps = [Xs_all_maps Xs_curr_map];
                end
                Xs_for_lr = [Xs_for_lr; Xs_all_maps];
            end
        end
        all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr ones(size(Xs_for_lr, 1),1)*cnt_sbj];
        all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr];
    end

    tbl_to_fit = array2table( ...
        [all_Ys_for_lr, all_Xs_for_lr], ...
        "VariableNames", ["choice", "CxR_S", "CxR_C", "CxR_P", "CxR_PC", "CxR_PS", "CxR_SC", "subject"]);
    mdl = fitglme(tbl_to_fit, 'choice~CxR_S+CxR_C+CxR_P+CxR_PC+CxR_PS+CxR_SC+(CxR_S+CxR_C+CxR_P+CxR_PC+CxR_PS+CxR_SC|subject)', 'Distribution','Binomial')
    mdls{cnt_O} = mdl;
end

%%

for i=1:27
    bs(i,:) = mdls{i}.Coefficients.Estimate;
end
figure
imagesc(mean(bs, 3))
caxis([-1, 1])
colormap bluewhitered
pbaspect([1 1 1])

all_maps_sim = { ...
    (colorMap(:)'==colorMap(:))-eye(27),...
    (shapeMap(:)'==shapeMap(:))-eye(27),...
    (patternMap(:)'==patternMap(:))-eye(27),...
    (patternshapeMap(:)'==patternshapeMap(:))-eye(27),...
    (patterncolorMap(:)'==patterncolorMap(:))-eye(27),...
    (shapecolorMap(:)'==shapecolorMap(:))-eye(27),...
    eye(27)};

all_maps_sim_flat = [];
for i=1:7
    all_maps_sim_flat(:,i) = all_maps_sim{i}(:);
end

mdl = fitlm(all_maps_sim_flat, reshape(mean(bs, 3), [], 1), ...
    'VarNames', {'F_inf', 'F_noninf1', 'F_noninf2', 'C_inf', 'C_noninf1', 'C_noninf2', 'O', 'Weight'})
