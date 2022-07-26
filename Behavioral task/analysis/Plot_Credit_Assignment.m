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
wSize  = 20 ;

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
% C_t ~ sum_i [beta_1 (C_{t-i}Xrew_{t-i}-C_{t-i}Xunr_{t-i})]
% each term coded as binary variable:
%   1. chosen and rewarded,
%   2. chosen and unrewarded,
% all of them are zero if that instance not present during that trial
% do separately for each feature and each conjunction (each object?)
% two types of generalization: see feature, change conjunction
%                           or see conjunction, change feature

lags_to_fit = 3; % how far back should the CxR variables be included

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

clear all_Xs_for_lr all_Ys_for_lr lr_weights_mean
for i_dim = 4:6
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

    if i_dim<=3
        for dim_inst = 1:max(obj2dim_map,[],'all')
            curr_dim_inst = cumsum([0 3 3 3 9 9 9]);
            curr_dim_inst = curr_dim_inst(i_dim) + dim_inst;

            all_Xs_for_lr = [];
            all_Ys_for_lr = [];

            for cnt_sbj = 1:length(subjects_inputs)
                if mod(cnt_sbj, 10)==0
                    disp(cnt_sbj)
                end

                inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
                resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

                inputs_struct = load(inputname);
                results_struct = load(resultsname);

                ntrialPerf = 33:length(results_struct.results.reward);
                [~, idxMax] = max(results_struct.expr.prob{1}(inputs_struct.input.inputTarget)) ;
                choiceRew{cnt_sbj} = results_struct.results.choice' == idxMax;
                perfMean(cnt_sbj) = mean(choiceRew{cnt_sbj}(ntrialPerf)) ;

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

                Ct = obj2dim_map(targCh)==dim_inst;
                uCt = obj2dim_map(targUnch)==dim_inst;

                % discard all trials where the feat/conj instance is not present
                rows_to_include = Ct==1 | uCt==1;
                Ct = Ct(rows_to_include);
                uCt = uCt(rows_to_include);

                % chosen and rewarded
                CtxRew = idx_rew(rows_to_include)*Ct';
                % chosen and unrewarded
                CtxUnr = idx_unr(rows_to_include)*Ct';
                Unch = uCt';

                Ys_for_lr = Ct(2:end)==Ct(1:end-1);
                Xs_for_lr = CtxRew(1:end-1)-CtxUnr(1:end-1);
                %6:end ~ 5:end-1Xs
                for j_lag=1:lags_to_fit
                    Xs_for_lr = [Xs_for_lr ...
                        CtxRew(lags_to_fit-j_lag+1:end-j_lag)-CtxUnr(lags_to_fit-j_lag+1:end-j_lag)];
                end
                all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr nominal(ones(size(Xs_for_lr, 1),1)*cnt_sbj)];
                all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr'];
            end
            tbl_to_fit = array2table( ...
                [all_Ys_for_lr, all_Xs_for_lr], ...
                "VariableNames", ["choice", "CxR1", "CxR2", "CxR3", "CxR4", "CxR5", "subject"]);
            mdl = fitglme(tbl_to_fit, ...
                'choice~CxR1+CxR2+CxR3+CxR4+CxR5+(CxR1+CxR2+CxR3+CxR4+CxR5|subject)', ...
                'Distribution','Binomial');
            %         lr_weights(dim_inst, :) = mdl.CoefficientCovariance^(-1)*mdl.Coefficients.Estimate;
            %         lr_weights_cov_inv(dim_inst,:, :) = mdl.CoefficientCovariance^(-1);
            mdls{curr_dim_inst} = mdl;
        end
        % lr_weights_mean_conj = lr_weights_mean_conj(idxperf, :, :);
    end
%     else
%         for dim_inst = 1:max(obj2dim_map,[],'all')
% 
%             curr_dim_inst = cumsum([0 3 3 3 9 9 9]);
%             curr_dim_inst = curr_dim_inst(i_dim) + dim_inst;
% 
%             all_Xs_for_lr = [];
%             all_Ys_for_lr = [];
% 
%             for cnt_sbj = 1:length(subjects_inputs)
%                 if mod(cnt_sbj, 10)==0
%                     disp(cnt_sbj)
%                 end
% 
%                 inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
%                 resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;
% 
%                 inputs_struct = load(inputname);
%                 results_struct = load(resultsname);
% 
%                 ntrialPerf = 33:length(results_struct.results.reward);
%                 [~, idxMax] = max(results_struct.expr.prob{1}(inputs_struct.input.inputTarget)) ;
%                 choiceRew{cnt_sbj} = results_struct.results.choice' == idxMax;
%                 perfMean(cnt_sbj) = mean(choiceRew{cnt_sbj}(ntrialPerf)) ;
% 
%                 inputTarget   = inputs_struct.input.inputTarget;
%                 Ntrials      = length(results_struct.results.reward);
%                 rewards      = results_struct.results.reward;
%                 choices      = results_struct.results.choice;
% 
%                 % the object chosen and unchosen
%                 targCh = inputs_struct.input.inputTarget(...
%                     results_struct.results.choice'+2*(0:(results_struct.expr.Ntrials-1))) ;
%                 targUnch = inputs_struct.input.inputTarget(...
%                     (3-results_struct.results.choice)'+2*(0:(results_struct.expr.Ntrials-1))) ;
% 
%                 % whether rewarded or not
%                 idx_rew = rewards==1;
%                 idx_unr = rewards==0;
% 
%                 Ct_ft_1 = obj2ft1_map(targCh)==ceil(dim_inst/3);
%                 Ct_ft_2 = obj2ft2_map(targCh)==mod(dim_inst-1, 3)+1;
%                 Ct_conj = obj2dim_map(targCh)==dim_inst;
%                 uCt_ft_1 = obj2ft1_map(targUnch)==ceil(dim_inst/3);
%                 uCt_ft_2 = obj2ft2_map(targUnch)==mod(dim_inst-1, 3)+1;
%                 uCt_conj = obj2dim_map(targUnch)==dim_inst;
%     
%                 % discard all trials where the feat/conj instance is not present
%                 rows_to_include = ...
%                     Ct_ft_1==1 | Ct_ft_2==1 | ...
%                     uCt_ft_1==1 | uCt_ft_2==1;
%                 Ct_ft_1 = Ct_ft_1(rows_to_include);
%                 Ct_ft_2 = Ct_ft_2(rows_to_include);
%                 Ct_conj = Ct_conj(rows_to_include);
%                 uCt_ft_1 = uCt_ft_1(rows_to_include);
%                 uCt_ft_2 = uCt_ft_2(rows_to_include);
%                 uCt_conj = uCt_conj(rows_to_include);
%     
%                 % chosen and rewarded
%                 CtxRew_ft_1 = idx_rew(rows_to_include).*(Ct_ft_1==1|uCt_ft_1==1)';
%                 CtxRew_ft_2 = idx_rew(rows_to_include).*(Ct_ft_2==1|uCt_ft_2==1)';
%                 CtxRew_conj = idx_rew(rows_to_include).*(Ct_conj==1|uCt_conj==1)';
%                 % chosen and unrewarded
%                 CtxUnr_ft_1 = idx_unr(rows_to_include).*(Ct_ft_1==1|uCt_ft_1==1)';
%                 CtxUnr_ft_2 = idx_unr(rows_to_include).*(Ct_ft_2==1|uCt_ft_2==1)';
%                 CtxUnr_conj = idx_unr(rows_to_include).*(Ct_conj==1|uCt_conj==1)';
%     
%                 Xs_for_lr = [CtxRew_ft_1-CtxUnr_ft_1,...
%                              CtxRew_ft_2-CtxUnr_ft_2,...
%                              CtxRew_conj-CtxUnr_conj]; % 1-T
%                 Xs_for_lr = Xs_for_lr(1:end, :);
% 
%                 rows_to_include = Ct_conj(2:end)==1 | uCt_conj(2:end)==1;
%                 % select trials whose next options include the conjunction
% 
%                 Xs_for_lr = Xs_for_lr(rows_to_include, :); % 1-t-1
% 
%                 Ct_conj = Ct_conj(rows_to_include); % 1-t
%                 % select trials whose current options include the
%                 % conjunction
%                 
%                 Ys_for_lr = Ct_conj(2:end)==Ct_conj(1:end-1); % 2-t
%                 Xs_for_lr = Xs_for_lr(1:end-1,:); % 1-t-1
% 
% 
%                 all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr ones(size(Xs_for_lr, 1),1)*cnt_sbj];
%                 all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr'];
%             end
%             tbl_to_fit = array2table( ...
%                 [all_Ys_for_lr, all_Xs_for_lr], ...
%                 "VariableNames", ["stay", "CxR_1", "CxR_2", "CxR_12", "subject"]);
%             mdl = fitglme(tbl_to_fit, ...
%                 'stay~CxR_1+CxR_2+CxR_12+(CxR_1+CxR_2+CxR_12|subject)', ...
%                 'Distribution','Binomial');
%             mdls{curr_dim_inst} = mdl;
    end
end