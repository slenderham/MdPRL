clc
close all
clear

set(0,'defaultAxesFontSize',30)

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
idxperf = find(idxperf);
% idxperf(36) = false;


%%

probEst_trials = [86, 173, 259, 346, 432];

all_probe_est_values = [];

for cnt_probe = 1:length(results.probEst)
    all_RL_values = [];
    all_est_values = [];
    for cnt_sbj = 1:length(idxperf)
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

        vf = curr_subj_vals(1:9,probEst_trials(cnt_probe));
        vc = curr_subj_vals(10:36,probEst_trials(cnt_probe));
        
        idx_shape    = shapeMap(:) ; % 1-3
        idx_color    = colorMap(:)+3 ; % 4-6
        idx_pattern  = patternMap(:)+6 ; % 7-9
        idx_patternshape = (idx_pattern-7)*3 + idx_shape; % 1-9
        idx_patterncolor = (idx_pattern-7)*3 + (idx_color-4)+10 ; % 10-18
        idx_shapecolor = (idx_shape-1)*3 + (idx_color-4)+19 ; % 19-27

        omega = attns.fit_results{5, 3, idxperf(cnt_sbj)}.params(3);
        beta = attns.fit_results{5, 3, idxperf(cnt_sbj)}.params(2);

        values(:, 1) = vf(idx_shape);
        values(:, 2) = vf(idx_color);
        values(:, 3) = vf(idx_pattern);
        values(:, 4) = vc(idx_patterncolor);
        values(:, 5) = vc(idx_patternshape);
        values(:, 6) = vc(idx_shapecolor);
        values(:, 7) = (omega*mean(values(:, 1:3), 2)+(1-omega)*mean(values(:, 4:6), 2));

        probEst = results.probEst{cnt_probe}(:) ;
        probEst = probEst./(1+probEst) ;
        probEst = round(probEst./0.05)*0.05 ;
        if probEst<0.05
            probEst = 0.05;
        elseif probEst>0.95
            probEst = 0.95;
        end
        probEst = log(probEst+1e-6) - log(1-probEst+1e-6);
        probEst(isnan(probEst)) = 0;
        all_RL_values = [all_RL_values; zscore(values) ones(size(values, 1),1)*cnt_sbj];
        all_est_values = [all_est_values; probEst];
    end
    tbl = array2table([all_est_values all_RL_values], ...
        "VariableNames", ["estimation", "S", "C", "P", "PC", "PS", "SC", "all", "subj"]);
    mdls{cnt_probe, 1} = fitlme(tbl, "estimation~S+C+P+PC+PS+SC+(S+C+P+PC+PS+SC|subj)",...
        "CovariancePattern", "Diagonal", "Verbose", 0,'CheckHessian',true);
    mdls{cnt_probe, 2} = fitlme(tbl, "estimation~all+(all|subj)",...
        'CovariancePattern','FullCholesky',"Verbose", 0, 'CheckHessian',true);
    all_probe_est_values(cnt_probe, :) = all_est_values;
end

%%

pe_mdls = load('../files/prob_est_models.mat');

for cnt_probe = 1:length(results.probEst)
%     rsqs(cnt_probe, 1) = mdls{cnt_probe, 1}.Rsquared.Adjusted;
%     bs(cnt_probe, :) = mdls{cnt_probe,1}.Coefficients.Estimate([3 2 4 6 5 7]);
%     bse(cnt_probe, :) = mdls{cnt_probe,1}.Coefficients.SE([3 2 4 6 5 7]);
    bs_uw(cnt_probe) = mdls{cnt_probe,2}.Coefficients.Estimate(2);
    bse_uw(cnt_probe) = mdls{cnt_probe,2}.Coefficients.SE(2);
    for cnt_mdl = 1:5
        rsqs(cnt_probe, cnt_mdl) = pe_mdls.mdls{cnt_mdl, cnt_probe}.Rsquared.Adjusted;
    end
    rsqs(cnt_probe, 6) = mdls{cnt_probe, 2}.Rsquared.Adjusted;
end

%%
figure
cmap = colormap('turbo(6)');
cmap(6,:) = 0.5;
for i=1:6
    plot(rsqs(:,i), 'Color', cmap(i,:), 'LineWidth', 1, "Marker", "o");hold on
end
legend(["F_{inf}", "F_{inf}+C_{inf}", "C_{noninf1}", "C_{noninf2}", "O", "RL"], ...
    "Location", "eastoutside", 'Orientation', 'vertical');
xlim([0.75 5.25])
ylim([0., 0.51])
xticks(1:5)
ylabel('Adj. R^2')
xlabel('Value estimation bout')
box off

%%
figure
% errorbar(bs, bse,  "Marker", 'o', 'LineWidth', 0.1, 'LineStyle', ':', 'CapSize', 0); hold on
errorbar(bs_uw, bse_uw, '-o', 'MarkerSize', 5, 'LineWidth', 1, 'Color', 0.5*ones(3,1))
xlim([0.5 5.5])
ylim([0.3 0.6])
yticks(0:0.1:1)
legend(["RL"])
xlabel('Value Estimation Trial')
ylabel('Regression Weights')
% plot(bs_uw/6, 'k')
% patch([1:5 5:-1:1], [bs_uw/6-bse_uw/6 fliplr(bs_uw/6+bse_uw/6)], 'k', 'FaceAlpha', 0.1)
