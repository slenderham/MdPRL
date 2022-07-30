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

clear all_Xs_for_lr all_Ys_for_lr lr_weights_mean

for cnt_O_fb = 1:27
    for cnt_O_ch = 1:27
        all_Xs_for_lr = [];
        all_Ys_for_lr = [];
        disp(['Feedback on ', num2str(cnt_O_fb), ', choice on ', num2str(cnt_O_ch)]);
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

            % whether rewarded or not
            idx_rew = rewards==1;
            idx_unr = rewards==0;

            % trials where either object is appears
            Ct_fb = targCh==cnt_O_fb; 
            Ct_ch = targCh==cnt_O_ch; 
            uCt_fb = targUnch==cnt_O_fb;
            uCt_ch = targUnch==cnt_O_ch;

            % discard all trials where the object
            rows_to_include_fb = Ct_fb==1 | uCt_fb==1;
            rows_to_include_ch = Ct_ch==1 | uCt_ch==1;
            rows_to_include = rows_to_include_fb | rows_to_include_ch;
            
            rows_to_include_fb = rows_to_include_fb(rows_to_include);
            rows_to_include_ch = rows_to_include_ch(rows_to_include);
            
            Ct_fb = Ct_fb(rows_to_include);
            uCt_fb = uCt_fb(rows_to_include);
            
            Ct_ch = Ct_ch(rows_to_include);

            % chosen and rewarded
            CtxRew = idx_rew(rows_to_include).*Ct_fb';
            % chosen and unrewarded
            CtxUnr = idx_unr(rows_to_include).*Ct_fb';
            Unch = uCt_fb';

            Xs_for_lr = [];
            Ys_for_lr = [];

            % for each time O_ch is an option
            % look back to find the last time O_fb is an option
            for l = 1:length(Ct_ch)
                if rows_to_include_ch(l)==1
                    for k=l-1:-1:1
                        if rows_to_include_fb(k)==1
                            Xs_for_lr = [Xs_for_lr; CtxRew(k)-CtxUnr(k)];
                            Ys_for_lr = [Ys_for_lr; Ct_ch(l)];
                            break;
                        end
                    end
                end
            end
            all_Xs_for_lr = [all_Xs_for_lr; Xs_for_lr ones(size(Xs_for_lr, 1),1)*cnt_sbj];
            all_Ys_for_lr = [all_Ys_for_lr; Ys_for_lr];
        end
        
        tbl_to_fit = array2table( ...
            [all_Ys_for_lr, all_Xs_for_lr], ...
            "VariableNames", ["choice", "CxR", "subject"]);
        mdl = fitglme(tbl_to_fit, 'choice~CxR+(CxR|subject)', 'Distribution','Binomial');
        mdls{cnt_O_fb, cnt_O_ch} = mdl;
    end
end

%% 

for i=1:27
    for j=1:27
        randEffects = randomEffects(mdls{i,j});
        bs(i,j,:) = mdls{i,j}.Coefficients.Estimate(2)+randEffects(2:2:end);
    end
end
figure
imagesc(mean(bs, 3))
caxis([-0.5, 0.5])
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

mdl = fitlm(all_maps_sim_flat, reshape(mean(bs), [], 1));
