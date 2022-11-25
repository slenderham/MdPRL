clc
close all
clear

set(0,'defaultAxesFontSize',22)

%%

probEst_trials = [86, 173, 259, 346, 432];

for cnt_sbj = 1:length(idxperf)
%     disp(strcat("Fitting subject ", num2str(cnt_sbj)));
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

    for cnt_probe = 1:length(results.probEst)
        vf = curr_subj_vals(1:9,probEst_trials(cnt_probe));
        vc = curr_subj_vals(10:36,probEst_trials(cnt_probe));
        
        idx_shape    = shapeMap(:) ; % 1-3
        idx_color    = colorMap(:)+3 ; % 4-6
        idx_pattern  = patternMap(:)+6 ; % 7-9
        idx_patternshape = (idx_pattern-7)*3 + idx_shape; % 1-9
        idx_patterncolor = (idx_pattern-7)*3 + (idx_color-4)+10 ; % 10-18
        idx_shapecolor = (idx_shape-1)*3 + (idx_color-4)+19 ; % 19-27

        omega = attns.fit_results{m, a, cnt_sbj}.params(3);

        values(cnt_sbj, cnt_probe, 1, :) = zscore(vf(idx_shape));
        values(cnt_sbj, cnt_probe, 2, :) = zscore(vf(idx_color));
        values(cnt_sbj, cnt_probe, 3, :) = zscore(vf(idx_pattern));
        values(cnt_sbj, cnt_probe, 4, :) = zscore(vf(idx_patterncolor));
        values(cnt_sbj, cnt_probe, 5, :) = zscore(vf(idx_patternshape));
        values(cnt_sbj, cnt_probe, 6, :) = zscore(vf(idx_shapecolor));

        all_rl_values(cnt_sbj, cnt_probe, :) = values;
        probEst = results.probEst{cnt_probe}(:) ;
        probEst = probEst./(1+probEst) ;
        probEst = round(probEst./0.05)*0.05 ;
        if probEst<0.05
            probEst = 0.05;
        elseif probEst>0.95
            probEst = 0.95;
        end
        all_est_values(cnt_sbj, cnt_probe, :) = probEst;
        [rho, p] = corr(values, results.probEst{cnt_probe}(:), 'type', 'spearman','rows','complete');
        value_prob_est_corrs(cnt_sbj, cnt_probe, :) = [rho, p];
    end
end
