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

%% logistic regression
% C_t ~ sum_i [beta_1 C_{t-i}Xrew_{t-i} + beta_2 C_{t-i}Xunr_{t-i}]
% each term coded as binary variable: 
%   1. chosen and rewarded, 
%   2. chosen and unrewarded,
% all of them are zero if that instance not present during that trial
% do separately for each feature and each conjunction (each object?)

perfTH = 0.53;
lags_to_fit = 5; % how far back should the CxR variables be included

clear Xs_for_lr lr_weights
for cnt_sbj = 1:length(subjects_inputs)

    disp(cnt_sbj)

    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

    inputs_struct = load(inputname);
    results_struct = load(resultsname);

    ntrialPerf = 33:length(results_struct.results.reward);
    [~, idxMax] = max(results_struct.expr.prob{1}(inputs_struct.input.inputTarget)) ;
    choiceRew{cnt_sbj} = results_struct.results.choice' == idxMax;
    perfMean(cnt_sbj) = mean(choiceRew{cnt_sbj}(ntrialPerf)) ;

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
    
    inputTarget   = inputs_struct.input.inputTarget;
    Ntrials      = length(results_struct.results.reward);
    rewards      = results_struct.results.reward;
    choices      = results_struct.results.choice;

    % the object chosen and unchosen
    targCh = inputs_struct.input.inputTarget(...
        results_struct.results.choice'+2*(0:(results_struct.expr.Ntrials-1))) ;
    targUnch = inputs_struct.input.inputTarget(...
        (3-results_struct.results.choice)'+2*(0:(results_struct.expr.Ntrials-1))) ;

    % whether rewarded or not
    idx_rew = rewards==1;
    idx_unr = rewards==0;

    for i_dim = 1:6
        switch i_dim
            case 1
                obj2dim_map = shapeMap;
            case 2
                obj2dim_map = colorMap;
            case 3
                obj2dim_map = patternMap;
            case 4
                obj2dim_map = patterncolorMap;
                obj2ft1_map = patternMap;
                obj2ft2_map = colorMap;
            case 5
                obj2dim_map = patternshapeMap;
                obj2ft1_map = patternMap;
                obj2ft2_map = shapeMap;
            case 6
                obj2dim_map = shapecolorMap;
                obj2ft1_map = shapeMap;
                obj2ft2_map = colorMap;
            otherwise
        end
        
        clear lr_weights lr_weights_cov_inv
        for dim_inst = 1:max(obj2dim_map,[],'all')
            curr_dim_inst = cumsum([0 3 3 3 9 9 9]);
            curr_dim_inst = curr_dim_inst(i_dim) + dim_inst;
            
            Ct = obj2dim_map(targCh)==dim_inst;
            uCt = obj2dim_map(targUnch)==dim_inst;

            % discard all trials where the feat/conj instance is not present
            rows_to_include = Ct==1 | uCt==1;
            Ct = Ct(rows_to_include);
            uCt = uCt(rows_to_include);
            
            % chosen and rewarded
            CtxRew = idx_rew(rows_to_include).*Ct';
            % chosen and unrewarded
            CtxUnr = idx_unr(rows_to_include).*Ct';
            Unch = uCt';
            
            Xs_for_lr = [];
            Ys_for_lr = Ct*1; 
            
            Ys_for_lr = Ys_for_lr(lags_to_fit+1:end);
            %6:end ~ 5:end-1
            for j_lag=1:lags_to_fit
                Xs_for_lr = [Xs_for_lr ...
                 CtxRew(lags_to_fit-j_lag+1:end-j_lag)-CtxUnr(lags_to_fit-j_lag+1:end-j_lag)];
            end
            % exclude if the previous few trials the instance is not chosen
%             rows_to_exclude = rows_to_exclude | sum(abs(Xs_for_lr), 2)'<1;
            [B, ~, stats] = glmfit(Xs_for_lr, Ys_for_lr', 'binomial', 'Options',statset('MaxIter',1000));
            lr_weights(dim_inst, :) = stats.covb^(-1)*B;
            lr_weights_cov_inv(dim_inst,:, :) = stats.covb^(-1);
        end
        lr_weights_mean(cnt_sbj, i_dim, :) = squeeze(sum(lr_weights_cov_inv, 1))^(-1)*sum(lr_weights, 1)';
    end
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
lr_weights_mean = lr_weights_mean(idxperf, :, :);