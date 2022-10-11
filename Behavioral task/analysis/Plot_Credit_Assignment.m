clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath('../utils')

set(0,'defaultAxesFontSize',18)
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
all_var_names = all_var_names + ["xR", "xNR"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);

all_probes = [0, 86, 173, 259, 346, 432];


%%
parfor cnt_probe = 1:length(all_probes)-1
    disp(num2str(cnt_probe));
    all_Xs_for_lr = [];
    all_Ys_for_lr = [];
    for cnt_O = 1:27    
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
            
            % for each subject, initialize array for recording previous rewards
            % and later choices
            Xs_for_lr = []; 
            Ys_for_lr = [];
    
            % for each time O_ch is an option
            % look back to find the last time O_fb is an option
            for l = all_probes(cnt_probe)+lags_to_fit+1:all_probes(cnt_probe+1)
                if (Ct(l) || uCt(l))  % if the object can be chosen at that trial
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
                        Xs_curr_map = [];
                        for k = l-1:-1:l-lags_to_fit % start from the previous trial and go back
                            if obj2dim_map(cnt_O)==obj2dim_map(targCh(k))
                                Xs_curr_map = [Xs_curr_map idx_rew(k) idx_unr(k)];
                            else
                                Xs_curr_map = [Xs_curr_map zeros(1,2)];
                            end
                        end
                        Xs_all_maps = [Xs_all_maps Xs_curr_map];
                    end
                    if true %sum(abs(Xs_all_maps))>0
                        Xs_for_lr = [Xs_for_lr; Xs_all_maps];
                        Ys_for_lr = [Ys_for_lr; Ct(l)];
                    end
                end
            end
            all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr ones(size(Xs_for_lr, 1),1)*cnt_sbj];
            all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr];
        end
    end
    tbl_to_fit = array2table( ...
        [all_Ys_for_lr, all_Xs_for_lr], ...
        "VariableNames", ["repeat", all_var_names', "subject"]);
    mdl = fitglme(tbl_to_fit, "repeat~"+strjoin(all_var_names, "+")+"+("+strjoin(all_var_names, "+")+"|subject)", ...
                  'Distribution','Binomial', 'CovariancePattern', 'Diagonal', 'FitMethod', 'Laplace');
    mdls{cnt_probe} = mdl;
end

cd ../files/
save("credit_assignment_models", "mdls")
cd ../analysis/

%%


all_var_names = ["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf1}"]';
all_var_names = all_var_names';
all_var_names = all_var_names(:);
all_var_names = all_var_names + ["xR", "xNR"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);

for j=1:length(all_probes)-1
    bs(j,:) = mdls{j}.Coefficients.Estimate(2:end);
    bse(j,:) = mdls{j}.Coefficients.SE(2:end);
end


bs = bs(:, [3 4 1 2 5 6 9 10 7 8 11 12]);
bse = bse(:, [3 4 1 2 5 6 9 10 7 8 11 12]);
figure
cmap_r = brighten(colormap('lines(6)'), 0.3);
cmap_d = brighten(colormap('lines(6)'), -0.3);
cmap = [];
for i=1:6
    cmap = [cmap; cmap_r(i,:); cmap_d(i,:)];
end
b = bar(bs');
hold on;
for i=1:5
b(i).FaceColor = 'flat';
b(i).CData = cmap;
end
xticks(1:12)
xticklabels(all_var_names)
xlim([0.5, 12.5])
xlabel('Variable')
ylabel('Regression Weight')
pbaspect([4 1 1])


[ngroups, nbars] = size(bs');
bx = nan(nbars, ngroups);
for i = 1:nbars
    bx(i,:) = b(i).XEndPoints;
end
% Plot the errorbars
e = errorbar(bx',bs',bse','k','linestyle','none','linewidth',0.01);
for i = 1:nbars
    e(i).CapSize = 1;
end
hold off


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
