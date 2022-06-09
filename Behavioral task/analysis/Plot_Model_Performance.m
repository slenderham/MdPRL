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

set(0,'defaultAxesFontSize',15)
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

    % calculate differential response
    expr.shapeMap       = expr.targetShape ;
    expr.colorMap       = expr.targetColor ;
    expr.patternMap     = expr.targetPattern ;
    
    input.inputTarget   = input.inputTarget(:, ntrialPerf) ;
    expr.Ntrials        = length(ntrialPerf) ;
    results.reward      = results.reward(ntrialPerf) ;
    results.choice      = results.choice(ntrialPerf) ;

    idx_rew             = find(results.reward(2:end-1)==1)+1 ;
    idx_unr             = find(results.reward(2:end-1)==0)+1 ;
    targCh              = input.inputTarget(results.choice'+2*(0:(expr.Ntrials-1))) ;
    
    pFtTrials{cnt_sbj}      = nan*ones(1, length(ntrialPerf)) ;
    pFt0Trials{cnt_sbj}     = nan*ones(1, length(ntrialPerf)) ;
    
    pObTrials{cnt_sbj}      = nan*ones(1, length(ntrialPerf)) ;
    pOb0Trials{cnt_sbj}     = nan*ones(1, length(ntrialPerf)) ;
    
    pFtinfTrials{cnt_sbj}   = nan*ones(1, length(ntrialPerf)) ;
    pFtinf0Trials{cnt_sbj}  = nan*ones(1, length(ntrialPerf)) ;
    
    pFtninfTrials{cnt_sbj}  = nan*ones(1, length(ntrialPerf)) ;
    pFtninf0Trials{cnt_sbj} = nan*ones(1, length(ntrialPerf)) ;
    
    pCjTrials{cnt_sbj}      = nan*ones(1, length(ntrialPerf)) ;
    pCj0Trials{cnt_sbj}     = nan*ones(1, length(ntrialPerf)) ;
    
    pCjnTrials{cnt_sbj}     = nan*ones(1, length(ntrialPerf)) ;
    pCjn0Trials{cnt_sbj}    = nan*ones(1, length(ntrialPerf)) ;
    
    for cnt_O = expr.playcombinations
        shapeO          = find(ismember(1:3, expr.shapeMap(cnt_O))) ;
        colorO          = find(ismember(1:3, expr.colorMap(cnt_O))) ;
        patternO        = find(ismember(1:3, expr.patternMap(cnt_O))) ;

        if expr.flaginf==1
            infO        = shapeO ;
            expr.inf    = expr.shapeMap ;
            cnt_uinf    = find(expr.patternMap==patternO | expr.colorMap==colorO) ;
            cnt_Tcj     = find(expr.patternMap==patternO & expr.colorMap==colorO) ;
            cnt_ucj     = find((expr.patternMap==patternO & expr.shapeMap==shapeO) | (expr.colorMap==colorO & expr.shapeMap==shapeO)) ;
        elseif expr.flaginf==2
            infO        = patternO ;
            expr.inf    = expr.patternMap ;
            cnt_uinf    = find(expr.shapeMap==shapeO | expr.colorMap==colorO) ;
            cnt_Tcj     = find(expr.shapeMap==shapeO & expr.colorMap==colorO) ;
            cnt_ucj     = find((expr.shapeMap==shapeO & expr.patternMap==patternO) | (expr.colorMap==colorO & expr.patternMap==patternO)) ;
        end
        
        % get all objects that share at least one feature
        cnt_Tall        = find(expr.shapeMap==shapeO | expr.colorMap==colorO | expr.patternMap==patternO ) ;
        % remove the object itself
        cnt_Tall        = cnt_Tall(~ismember(cnt_Tall, cnt_O)) ;
        % find objects that share the informative feature
        cnt_Tinf        = find(expr.inf==infO) ;
        cnt_Tinf        = cnt_Tinf(~ismember(cnt_Tinf, cnt_O)) ; % remove itself
        % find objects that don't share the informative feature
        cnt_Tninf       = cnt_Tall(~ismember(cnt_Tall, cnt_Tinf)) ; 
        cnt_Tinf        = cnt_Tinf(~ismember(cnt_Tinf, cnt_uinf)) ;
        % find objects that share the informative conjunction
        cnt_Tcj         = cnt_Tcj(~ismember(cnt_Tcj, cnt_O)) ;
        % find objects that share the uninformative conjunction
        cnt_ucj         = cnt_ucj(~ismember(cnt_ucj, cnt_O)) ;
        cnt_Tninf       = cnt_Tninf(~ismember(cnt_Tninf, cnt_Tcj)) ;
        cnt_Trest       = find(expr.shapeMap~=shapeO & expr.colorMap~=colorO & expr.patternMap~=patternO ) ;
        
        % 1) inf feature, 2) noninf feature, 3) inf conj, 4) object
        idxCell{1}      = cnt_Tinf ;
        idxCell{2}      = cnt_Tninf ;
        idxCell{3}      = cnt_Tcj ;
        idxCell{4}      = cnt_ucj ;
        idxCell{5}      = cnt_O ;
        
        % find trials with reward on object in trial (i) and no object on (i+1)
        idx_rewO        = idx_rew(targCh(idx_rew)==cnt_O & ...
            sum(ismember(input.inputTarget(:, idx_rew+1),  cnt_O))==0 & ...
            sum(ismember(input.inputTarget(:, idx_rew+1),  cnt_Trest))==1) ;
        idx_unrO        = idx_unr(targCh(idx_unr)==cnt_O & ...
            sum(ismember(input.inputTarget(:, idx_unr+1),  cnt_O))==0 & ...
            sum(ismember(input.inputTarget(:, idx_unr+1),  cnt_Trest))==1) ;
        
        % find trials with no target and no similar features for both options on trial (i+1):
        idx_rewOI       = idx_rewO(sum(ismember(input.inputTarget(:, idx_rewO+1),  cnt_Tinf))==1) ;
        idx_unrOI       = idx_unrO(sum(ismember(input.inputTarget(:, idx_unrO+1),  cnt_Tinf))==1) ;
        
        idx_rewON       = idx_rewO(sum(ismember(input.inputTarget(:, idx_rewO+1),  cnt_Tninf))==1) ;
        idx_unrON       = idx_unrO(sum(ismember(input.inputTarget(:, idx_unrO+1),  cnt_Tninf))==1) ;
        
        idx_rewOT       = idx_rewO(sum(ismember(input.inputTarget(:, idx_rewO+1),  cnt_Tall))==1) ;
        idx_unrOT       = idx_unrO(sum(ismember(input.inputTarget(:, idx_unrO+1),  cnt_Tall))==1) ;
        
        idx_rewOC       = idx_rewO(sum(ismember(input.inputTarget(:, idx_rewO+1),  cnt_Tcj))==1) ;
        idx_unrOC       = idx_unrO(sum(ismember(input.inputTarget(:, idx_unrO+1),  cnt_Tcj))==1) ;
        
        idx_rewIC       = idx_rewO(sum(ismember(input.inputTarget(:, idx_rewO+1),  cnt_ucj))==1) ;
        idx_unrIC       = idx_unrO(sum(ismember(input.inputTarget(:, idx_unrO+1),  cnt_ucj))==1) ;
        
        % find trials with reward on object in trial (i) and object on (i+1)
        idx_rewOb       = idx_rew(targCh(idx_rew)==cnt_O & sum(ismember(input.inputTarget(:, idx_rew+1),  cnt_O))==1) ;
        idx_unrOb       = idx_unr(targCh(idx_unr)==cnt_O & sum(ismember(input.inputTarget(:, idx_unr+1),  cnt_O))==1) ;
        
        % find trials with target and no similar features for both options on trial (i+1):
        idx_rewOb       = idx_rewOb(sum(ismember(input.inputTarget(:, idx_rewOb+1),  cnt_Tall))==0) ;
        idx_unrOb       = idx_unrOb(sum(ismember(input.inputTarget(:, idx_unrOb+1),  cnt_Tall))==0) ;
        
        % save through time
        pFtTrials{cnt_sbj}(idx_rewOT+1)         = ismember(targCh(idx_rewOT+1), cnt_Tall) ;
        pFt0Trials{cnt_sbj}(idx_unrOT+1)        = ismember(targCh(idx_unrOT+1), cnt_Tall) ;
        
        pObTrials{cnt_sbj}(idx_rewOb+1)         = ismember(targCh(idx_rewOb+1), cnt_O) ;
        pOb0Trials{cnt_sbj}(idx_unrOb+1)        = ismember(targCh(idx_unrOb+1), cnt_O) ;
        
        pFtinfTrials{cnt_sbj}(idx_rewOI+1)      = ismember(targCh(idx_rewOI+1), cnt_Tinf) ;
        pFtinf0Trials{cnt_sbj}(idx_unrOI+1)     = ismember(targCh(idx_unrOI+1), cnt_Tinf) ;
        
        pFtninfTrials{cnt_sbj}(idx_rewON+1)     = ismember(targCh(idx_rewON+1), cnt_Tninf) ;
        pFtninf0Trials{cnt_sbj}(idx_unrON+1)    = ismember(targCh(idx_unrON+1), cnt_Tninf) ;
        
        pCjTrials{cnt_sbj}(idx_rewOC+1)         = ismember(targCh(idx_rewOC+1), cnt_Tcj) ;
        pCj0Trials{cnt_sbj}(idx_unrOC+1)        = ismember(targCh(idx_unrOC+1), cnt_Tcj) ;
        
        pCjnTrials{cnt_sbj}(idx_rewIC+1)        = ismember(targCh(idx_rewIC+1), cnt_ucj) ;
        pCjn0Trials{cnt_sbj}(idx_unrIC+1)       = ismember(targCh(idx_unrIC+1), cnt_ucj) ;
    end
    pFtAll(cnt_sbj)     = nanmean(pFtTrials{cnt_sbj})    - nanmean(pFt0Trials{cnt_sbj}) ;
    pObAll(cnt_sbj)     = nanmean(pObTrials{cnt_sbj})    - nanmean(pOb0Trials{cnt_sbj}) ;
    pFtinfAll(cnt_sbj)  = nanmean(pFtinfTrials{cnt_sbj}) - nanmean(pFtinf0Trials{cnt_sbj}) ;
    pFtninfAll(cnt_sbj) = nanmean(pFtninfTrials{cnt_sbj})- nanmean(pFtninf0Trials{cnt_sbj}) ;
    pCjinfAll(cnt_sbj)  = nanmean(pCjTrials{cnt_sbj})    - nanmean(pCj0Trials{cnt_sbj}) ;
    pCjninfAll(cnt_sbj) = nanmean(pCjnTrials{cnt_sbj})   - nanmean(pCjn0Trials{cnt_sbj}) ;
    
    pFt(cnt_sbj,1)     = nanmean(pFtTrials{cnt_sbj}) ;
    pOb(cnt_sbj,1)     = nanmean(pObTrials{cnt_sbj}) ;
    pFtinf(cnt_sbj,1)  = nanmean(pFtinfTrials{cnt_sbj}) ;
    pFtninf(cnt_sbj,1) = nanmean(pFtninfTrials{cnt_sbj}) ;
    pCjinf(cnt_sbj,1)  = nanmean(pCjTrials{cnt_sbj}) ;
    pCjninf(cnt_sbj,1) = nanmean(pCjnTrials{cnt_sbj}) ;
    
    pFt(cnt_sbj,2)     = nanmean(pFt0Trials{cnt_sbj}) ;
    pOb(cnt_sbj,2)     = nanmean(pOb0Trials{cnt_sbj}) ;
    pFtinf(cnt_sbj,2)  = nanmean(pFtinf0Trials{cnt_sbj}) ;
    pFtninf(cnt_sbj,2) = nanmean(pFtninf0Trials{cnt_sbj}) ;
    pCjinf(cnt_sbj,2)  = nanmean(pCj0Trials{cnt_sbj}) ;
    pCjninf(cnt_sbj,2) = nanmean(pCjn0Trials{cnt_sbj}) ;
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);

figure
plot_shaded_errorbar(mean(movmean(rew(idxperf,:), 20, 2))', std(movmean(rew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, 'k');
plot_shaded_errorbar(mean(movmean(choiceRew(idxperf,:), 20, 2))', std(movmean(choiceRew(idxperf,:), 20, 2))'/sqrt(length(idxperf)), 1, [0.5 0.5 0.5]);
xlabel('Trial Number')
ylabel('Performance')
legend({'', 'Reward', '', 'Proportion Better'})
xlim([0, 432])


%% load results with attn and ML params

attns = load('../files/RPL2Analysis_Attention_merged_rep40.mat') ;

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            lls(m, a, cnt_sbj) = attns.fit_results{m, a, idxperf(cnt_sbj)}.fval;            
            AICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+2*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
            BICs(m, a, cnt_sbj) = 2*lls(m, a, cnt_sbj)+log(ntrials)*length(attns.fit_results{m, a, idxperf(cnt_sbj)}.params);
        end
    end
end

%% bayesian model selection

[alpha_AIC,exp_r_AIC,xp_AIC,pxp_AIC,bor_AIC,g_AIC] = bms(reshape(-permute(AICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_AIC);
figure;
h = bar(reshape(pxp_AIC, 10, 5));hold on;

hx_pos = nan(10, 5);
for i = 1:5
    hx_pos(:,i) = h(i).XEndPoints;
end

plot(hx_pos(:), alpha_AIC/sum(alpha_AIC), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()


[alpha_BIC,exp_r_BIC,xp_BIC,pxp_BIC,bor_BIC,g_BIC] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell((1:50)', repmat([1], 1, 50)));
disp(bor_BIC);
figure;
h = bar(reshape(pxp_BIC, 10, 5));hold on;

hx_pos = nan(10, 5);
for i = 1:5
    hx_pos(:,i) = h(i).XEndPoints;
end

plot(hx_pos(:), alpha_BIC/sum(alpha_BIC), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend)
set(h, {'DisplayName'}, {'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'}')
ylabel('pxp')
legend()

[~, best_model_inds] = max(g_BIC);
% title('Posterior Model Probability')

[alpha_input,exp_r_input,xp_input,pxp_input,bor_input,g_input] = bms(reshape(-permute(BICs/2, [2 1 3]), [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [10, 5])', repmat([1], 1, 5)));
disp(bor_input);
figure;
h = bar(pxp_input);hold on;
plot(alpha_input/sum(alpha_input), 'ko', 'DisplayName', "E[Freq]")
xticks(1:5)
xticklabels({'F', 'F+O', 'F+C_{untied}', 'F+C_{feat attn}', 'F+C_{tied}'})
ylabel('pxp')

[alpha_attn,exp_r_attn,xp_attn,pxp_attn,bor_attn,g_attn] = bms(reshape(-BICs/2, [50, length(idxperf)])', ...
                                mat2cell(reshape(1:50, [5, 10])', repmat([1], 1, 10)));
disp(bor_attn);
figure;
h = bar(pxp_attn);hold on;
plot(alpha_attn/sum(alpha_attn), 'ko', 'DisplayName', "E[Freq]")
xticks(1:10)
xticklabels(attn_modes_legend(:,1))
ylabel('pxp')

%% plot DRs

range = [-0.6:0.1:0.6] ;

figure
hold on
plot(pFtninfAll(idxperf), pFtinfAll(idxperf), 'db', 'linewidth', 2, 'markersize', 8)
plot(-0.6:0.1:0.6, -0.6:0.1:0.6, '--k')
axis([-0.6 1.2 -0.6 0.6])
set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','normal','LineWidth',2,'yTick',-0.6:0.2:0.6,'Xtick',-0.60:0.2:0.6)
box off
set(gca, 'tickdir', 'out')
ylabel('DR inf. feature')
xlabel('DR non inf. features')

axes('position', [0.65 0.25 0.2 0.2])
hold on
histogram(pFtinfAll(idxperf)-pFtninfAll(idxperf), range, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7) ;
plot(nanmedian(pFtinfAll(idxperf)-pFtninfAll(idxperf))*ones(10,1), linspace(0,19.6, 10),'--b', 'linewidth', 1)
plot(0*ones(10,1), linspace(0,19.6, 10),'-', 'color', 0*[1 1 1], 'linewidth', 1)
set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','normal','LineWidth',2,'yTick',0:10:20,'Xtick',-0.6:0.6:0.6)
box off
set(gca, 'tickdir', 'out')
axis([-0.6 0.6 0 20])


figure
hold on
plot(pCjninfAll(idxperf(best_model_inds>20)), pCjinfAll(idxperf(best_model_inds>20)), 'db', 'linewidth', 2, 'markersize', 8)
plot(-0.6:0.1:0.6, -0.6:0.1:0.6, '--k')
axis([-0.6 1.2 -0.6 0.6])
set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','normal','LineWidth',2,'yTick',-0.6:0.2:0.6,'Xtick',-0.60:0.2:0.6)
box off
set(gca, 'tickdir', 'out')

ylabel('DR inf. conjunction')
xlabel('DR non inf. conjunctions')

axes('position', [0.65 0.25 0.2 0.2])
hold on
histogram(pCjinfAll(idxperf(best_model_inds>20))-pCjninfAll(idxperf(best_model_inds>20)), range, 'FaceColor', 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.7) ;
plot(nanmedian(pCjinfAll(idxperf(best_model_inds>20))-pCjninfAll(idxperf(best_model_inds>20)))*ones(10,1), linspace(0,19.6, 10),'--b', 'linewidth', 1)
plot(0*ones(10,1), linspace(0,19.6, 10),'-', 'color', 0*[1 1 1], 'linewidth', 1)
set(gca,'FontName','Helvetica','FontSize',20,'FontWeight','normal','LineWidth',2,'yTick',0:10:20,'Xtick',-0.6:0.6:0.6)
box off
set(gca, 'tickdir', 'out')
axis([-0.6 0.6 0 20])


%% Simulate model with best param

for m = 1:length(all_model_names)
    disp("=======================================================");
    disp(strcat("Fitting model ", all_model_names(m)));
    basic_params = cell(length(subjects_inputs)); % store the attention-less model's parameters for each model type
    for a = 1:length(attn_modes)
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_sbj = 1:length(subjects_inputs)
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

            ll = str2func(all_model_names(m));
            
            [trial_ll, trial_values, trial_attns] = ll(best_pars, sesdata);

            all_values{m, a, cnt_sbj} = trial_values;
            all_attns{m, a, cnt_sbj} = trial_attns;
            trial_lls(m, a, cnt_sbj, :) = trial_ll;
            trial_AICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+2*length(best_pars)/ntrials;
            trial_BICs(m, a, cnt_sbj, :) = 2*trial_lls(m, a, cnt_sbj, :)+log(ntrials)*length(best_pars)/ntrials;
        end
    end
end

unfilt_AICs = trial_AICs;
unfilt_BICs = trial_BICs;

trial_AICs=trial_AICs(:,:,idxperf,:);
trial_BICs=trial_BICs(:,:,idxperf,:);

%% Plot Model Evidence

wSize = 50;
clrmat = colormap('lines(5)') ;

smth_AIC = movmean(trial_AICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_BIC = movmean(trial_BICs, [0 wSize-1], 4, 'Endpoints', 'discard');
smth_ll = movmean(trial_lls, [0 wSize-1], 4, 'Endpoints', 'discard');

for i=1:5
    [~, min_attn_type] = max(alpha_BIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
    disp(min_attn_type)
    l(i) = plot_shaded_errorbar(squeeze(median(smth_BIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(1.25*std(smth_BIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])
xlabel('Trial')
ylabel('Trial-wise BIC')

figure
for i=1:5
    [~, min_attn_type] = max(alpha_AIC((i-1)*length(attn_modes)+1:i*length(attn_modes)));
    disp(min_attn_type)
    l(i) = plot_shaded_errorbar(squeeze(median(smth_AIC(i,min_attn_type,:,:), [1 2 3])), ...
                                squeeze(1.25*std(smth_AIC(i,min_attn_type,:,:), [], [1 2 3]))/sqrt(length(idxperf)), ...
                                wSize, clrmat(i,:));hold on
end
legend(l, ["F", "F+O", "F+C_{feat attn}", "F+C_{untied}", "F+C_{tied}"])
xlabel('Trial')
ylabel('Trial-wise AIC')

%% compare evidence at diff epochs / detect transition

probeTrialsAll = load(['../PRLexp/inputs_all/inputs/input_', 'aa' , '.mat']).expr.trialProbe;
probeTrialsAll = [1 probeTrialsAll];

for i=1:length(probeTrialsAll)-1
    binned_trial_BICs(:,:,:,i) = mean(trial_BICs(:,:,:,probeTrialsAll(i):probeTrialsAll(i+1)), 4);
end


%% quantify effect of attention
%% sharpness of attention (entropy)? focus on the correct feature (cross entropy)

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            if strcmp(attn_modes(a,2), 'C')
                attn_where = 1;
            elseif strcmp(attn_modes(a,2), 'L')  
                attn_where = 2;
            elseif strcmp(attn_modes(a,2), 'CL')
                attn_where = 2;
            else
                attn_where = 1;
            end
            if m~=3
                all_model_ents(m, a, cnt_sbj, :) = squeeze(entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,:)));
                all_model_kls(m, a, cnt_sbj, :) = squeeze(symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,2:end), ...
                                                                      all_attns{m, a, idxperf(cnt_sbj)}(attn_where,:,1:end-1), 2));
            else
                all_model_ents(m, a, cnt_sbj, :) = squeeze((entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,:)) ...
                                                           +entropy(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,:)))/2);
                all_model_kls(m, a, cnt_sbj, :) = squeeze((symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,2:end), ...
                                                                       all_attns{m, a, idxperf(cnt_sbj)}(attn_where,1:3,1:end-1), 2) ...
                                                           +symm_kl_div(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,2:end), ...
                                                                       all_attns{m, a, idxperf(cnt_sbj)}(attn_where,4:6,1:end-1), 2))/2);
            end
        end
    end
end

% fig=  figure;
% tiledlayout(3,3);
% for a = 2:length(attn_modes)
%     nexttile;
%     for m = 1:length(all_model_names)
%         plot_shaded_errorbar(squeeze(mean(all_model_ents(m,a,:,:), 3)), squeeze(std(all_model_ents(m,a,:,:), [], 3))/sqrt(length(idxperf)), 1, clrmat(m,:));hold on;
%         ylim([0, 1.2])
%         if mod(a, 3)==2
%             ylabel(attn_modes(a, 1))
%         end
%         if (a-1)/3>2
%             xlabel(attn_modes(a, 2))
%         end
%     end
% end

% lg  = legend(reshape([repmat([""], 5, 1) all_model_names_legend(1,:)']', 10, 1)); 
% lg.Layout.Tile = 'East';
% fax=axes(fig,'visible','off'); 
% fax.XLabel.Visible='on';
% fax.YLabel.Visible='on';
% ylabel(fax,'Attn Type');
% xlabel(fax,'Attn Time');

figure;
posterior_model_ents = sum(all_model_ents.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
posterior_model_kls = sum(all_model_kls.*permute(reshape(g_BIC, [10 5 length(idxperf)]), [2 1 3]), [1 2]);
plot_shaded_errorbar(squeeze(mean(posterior_model_ents, 3)), squeeze(std(posterior_model_ents, [], 3))/sqrt(length(idxperf)), 1, rgb('deepskyblue'));hold on;
plot_shaded_errorbar(squeeze(mean(posterior_model_kls, 3)), squeeze(std(posterior_model_kls, [], 3))/sqrt(length(idxperf)), 1:ntrials-1, rgb('coral'));hold on;
legend(["", "Entropy", "", "KL_{symm}"])
ylim([0, 10])
xlim([0, ntrials+10])
xlabel('Trial')

%% figure out how the flaginf is used if at all

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            if strcmp(attn_modes(a,2), 'C')
                attn_where = 1;
            elseif strcmp(attn_modes(a,2), 'L')  
                attn_where = 2;
            elseif strcmp(attn_modes(a,2), 'CL')
                attn_where = 2;
            else
                attn_where = 1;
            end
            for d = 1:3
                if m~=3 
                    all_model_ces(m, a, cnt_sbj, :, d) = squeeze(all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:));
                else
                    conj_d = [2, 1, 3];
                    conj_d = conj_d(d);
                    all_model_ces(m, a, cnt_sbj, :, d) = squeeze((all_attns{m, a, idxperf(cnt_sbj)}(attn_where,d,:) ...
                                                               +all_attns{m, a, idxperf(cnt_sbj)}(attn_where,3+conj_d,:))/2);
                end
            end
        end
    end
end

% fig=  figure;
% tiledlayout(3,3);
% for a = 2:length(attn_modes)
%     nexttile;
%     for m = 1:length(all_model_names)
%         plot_shaded_errorbar(squeeze(mean(all_model_ces(m,a,:,:), 3)), squeeze(std(all_model_ces(m,a,:,:), [], 3))/sqrt(length(idxperf)), 1, clrmat(m,:));hold on;
%         ylim([0, 10])
%         if mod(a, 3)==2
%             ylabel(attn_modes(a, 1))
%         end
%         if (a-1)/3>2
%             xlabel(attn_modes(a, 2))
%         end
%     end
% end
% 
% lg  = legend(reshape([repmat([""], 5, 1) all_model_names_legend(1,:)']', 10, 1)); 
% lg.Layout.Tile = 'East';

figure
clrmat = colormap('winter(3)');
posterior_model_ces = sum(all_model_ces.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1]), [2 1 3 4 5]), [1 2]);
posterior_model_ces = squeeze(posterior_model_ces);
wSize = 20;
smth_ces = movmean(posterior_model_ces, [0 wSize-1], 2, 'Endpoints', 'discard');
for d=[2 1 3]
    plot_shaded_errorbar(squeeze(mean(smth_ces(:,:,d), 1))', squeeze(std(smth_ces(:,:,d), [], 1))'/sqrt(length(idxperf)), wSize, clrmat(d,:));hold on;
end
legend(["", "inf", "", "noninf1", "", "noninf2"]); 
ylim([0.1, 0.6])
xlim([wSize, ntrials+10])
xlabel('Trial')
ylabel('Attention Weights')

%% extra explained variance based on diff lls
figure;
imagesc(1-mean(lls(5,3,:), 3)./mean(lls, 3));
axis image
colorbar();
xticks(1:10)
xticklabels(attn_modes_legend)
yticks(1:5)
yticklabels(all_model_names_legend')

%% differential signals in value (pairwise difference in values in different dimensions)
%% significance of "complementary" attention

channel_groups = {[0 3 6 9], ...
                  [0 3 6 9 36], ...
                  [0 3 6 9 18 27 36], ...
                  [0 3 6 9 18 27 36], ...
                  [0 3 6 9 18 27 36]};

for m = 1:length(all_model_names)
    for a = 1:length(attn_modes)
        for cnt_sbj = 1:length(idxperf)
            for l = 1:ntrials
                cg = channel_groups{m};
                all_value_pdists(m, a, cnt_sbj, l, 1:6) = nan;
                for cg_idx = 1:length(cg)-1
                    all_value_pdists(m, a, cnt_sbj, l, cg_idx) = ...
                                 mean(pdist( ...
                                    all_values{m, a, idxperf(cnt_sbj)}(cg(cg_idx)+1:cg(cg_idx+1), l) ...
                                 ));
                end
            end
        end
    end
end

posterior_model_pdists = nansum(all_value_pdists.*permute(reshape(g_BIC, [10 5 length(idxperf) 1 1 1]), [2 1 3 4 5]), [1 2]);
posterior_model_pdists = squeeze(posterior_model_pdists);



%% model ablation to force attention on informative feature/conj pair

%% simulate without teacher-forcing trial to see robustness

%% run model recovery

%% look at parameter, confirmation bias?

%% differential response focus more informative conjunction
%% correlate differential signal with attention weights




%% **********
%% calculate value difference in attention-less model to see interaction between learning and attention
%% difference in parameters between attention and non-attention models (especially diff in lrs)
%% **********

%% **********
%% consistency of behavior with simulated models with attention to see if it accounts for 
%% **********


%% **********
%% register for sfn membership
%% **********