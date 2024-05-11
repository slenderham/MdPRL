clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath('../utils')

set(0,'defaultAxesFontSize',22)
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

ntrialPerf = 33:432;
perfTH = 0.53;

for cnt_sbj = 1:length(subjects_inputs)
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

all_var_names = ["S", "C", "P", "PC", "PS", "SC"]';
all_var_names = all_var_names';
all_var_names = all_var_names(:);
all_var_names = all_var_names + ["_Rw", "_Ch"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);

%% 

all_Xs_for_lr = [];
all_Ys_for_lr = [];
for cnt_sbj = 1:length(subjects_inputs)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

    inputs_struct = load(inputname);
    results_struct = load(resultsname);

    inputTarget   = inputs_struct.input.inputTarget;
    Ntrials      = length(results_struct.results.reward);
    rewards      = results_struct.results.reward;
    rts = results_struct.results.responsetime;

    targets = inputs_struct.input.inputTarget;
    % the object chosen and unchosen
    targCh = inputs_struct.input.inputTarget(...
        results_struct.results.choice'+2*(0:(results_struct.expr.Ntrials-1))) ;
    targUnch = inputs_struct.input.inputTarget(...
        (3-results_struct.results.choice)'+2*(0:(results_struct.expr.Ntrials-1))) ;

    % reward
    idx_rew = rewards==1;
    idx_unr = rewards==0;
    
    % for each subject, initialize array for recording previous rewards
    % and later choices
    Xs_for_lr = []; 
    Ys_for_lr = [];

    % for each time O_ch is an option
    % look back to find the last time O_fb is an option
    for l = 2:150
        if ismember(l, [86, 173, 259, 346]+1)
            continue
        end        
        Xs_all_maps = [l/432, idx_rew(l-1)];
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
            Xs_curr_map = [];
            for k = l-1:-1:l-lags_to_fit % start from the previous trial and go back
                if obj2dim_map(targCh(k))==obj2dim_map(targets(1,l))
                    Xs_curr_map = [Xs_curr_map 2*idx_rew(k)-1 1];
                elseif obj2dim_map(targCh(k))==obj2dim_map(targets(2,l))
                    Xs_curr_map = [Xs_curr_map 2*idx_rew(k)-1 1];
                else
                    Xs_curr_map = [Xs_curr_map zeros(1,2)];
                end
            end
            Xs_all_maps = [Xs_all_maps Xs_curr_map];
        end
        if true %sum(abs(Xs_all_maps))>0
            Xs_for_lr = [Xs_for_lr; Xs_all_maps];
            Ys_for_lr = [Ys_for_lr; log(rts(l))];
        end
    end
    all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr ones(size(Xs_for_lr, 1),1)*cnt_sbj];
    all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr];
end

tbl_to_fit = array2table( ...
    [all_Ys_for_lr, all_Xs_for_lr], ...
    "VariableNames", ["rts", "t", "Rw", all_var_names', "subject"]);
% mdl = fitlme(tbl_to_fit, "rts~t+Rw+"+strjoin(all_var_names, "+")+"+("+strjoin(all_var_names, "+")+"|subject)", ...
%               'CovariancePattern', 'Diagonal', 'Verbose', 0, 'CheckHessian', true)
mdl = fitlme(tbl_to_fit, "rts~t+Rw+"+strjoin(all_var_names, "+")+"+(1|subject)", ...
'CovariancePattern', 'Diagonal', 'Verbose', 0, 'CheckHessian', true)

% cd ../files/
% save("credit_assignment_models_diag_RwCh_first", "mdl")
% cd ../analysis/

%%

all_var_names = ["F_{0}", "F_{1}", "F_{2}", "C_{0}", "C_{1}", "C_{2}"]';
all_var_names = all_var_names';
all_var_names = all_var_names(:);
all_var_names = all_var_names + ["Rw", "Ch"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);


bs = mdl.Coefficients.Estimate(2:end);
bse = mdl.Coefficients.SE(2:end);


bs = bs([3 4 1 2 5 6 9 10 7 8 11 12]+2);
bs = reshape(bs', 2, 6);
bse = bse([3 4 1 2 5 6 9 10 7 8 11 12]+2);
bse = reshape(bse', 2, 6);
% bs = bs([2 1 3 5 4 6]);
% bse = bse([2 1 3 5 4 6]);
figure
cmap = colormap('lines(6)');
cmap = repelem(cmap,2,1);
b = bar(bs);
hold on;
xticklabels(["Reward", "Choice"])
% ylim([-0.15, 0.3])
xlabel('Variable')
ylabel('Regression Weight')
pbaspect([1.25, 1, 1])

x = nan(6, 2);
for i = 1:6
    x(i,:) = b(i).XEndPoints;
end

e = errorbar(x',bs,bse,'k','linestyle','none','linewidth',0.01);

legend(["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf2}",repmat([""],[1 6])], 'Location','eastoutside');


% text(x(1,1)-0.12, bs(1)+bse(1)+0.01, '****', 'FontSize', 30, 'Color', cmap(1,:))
% text(x(1,2)-0.07, bs(2)+bse(2)+0.01, '**', 'FontSize', 30, 'Color', cmap(2,:))
% text(x(2,2)-0.085, bs(4)+bse(4)+0.01, '***', 'FontSize', 30, 'Color', cmap(4,:))
% text(x(4,1)-0.035, bs(7)+bse(7)+0.01, '*', 'FontSize', 30, 'Color', cmap(7,:))


% all_maps_sim = { ...
%     (colorMap(:)'==colorMap(:))-eye(27),...
%     (shapeMap(:)'==shapeMap(:))-eye(27),...
%     (patternMap(:)'==patternMap(:))-eye(27),...
%     (patternshapeMap(:)'==patternshapeMap(:))-eye(27),...
%     (patterncolorMap(:)'==patterncolorMap(:))-eye(27),...
%     (shapecolorMap(:)'==shapecolorMap(:))-eye(27),...
%     eye(27)};
% 
% all_maps_sim_flat = [];
% for i=1:7
%     all_maps_sim_flat(:,i) = all_maps_sim{i}(:);
% end
% 
% mdl = fitlm(all_maps_sim_flat, reshape(mean(bs, 3), [], 1), ...
%     'VarNames', {'F_inf', 'F_noninf1', 'F_noninf2', 'C_inf', 'C_noninf1', 'C_noninf2', 'O', 'Weight'})
