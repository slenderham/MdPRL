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

set(0,'defaultAxesFontSize',25)

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
end

idxperf = perfMean>=perfTH;
idxperf(29) = 0;
idxperf = find(idxperf);
% idxperf = 1:length(subjects);


%%

num_blocks = 4;

for idx_block = 1:num_blocks
    ntriaToFit = 432/num_blocks*(idx_block-1)+1:432/num_blocks*idx_block;

    all_diff_vals = [];
    all_choicetrials = [];
    all_option_pairs = [];
    
    all_choice_probs = {[], []};
    
    pair_to_ind = [nan 1 2; 3 nan 4; 5 6 nan];
    
    clear diff_vals
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
    
        
        % shape, color, pattern
        vf = [squeeze(mean(expr.prob{1}, [1 3])) ...
              squeeze(mean(expr.prob{1}, [2 3]))' ...
              squeeze(mean(expr.prob{1}, [1 2]))']; 
    
        % patternshape patterncolor shapecolor
        vc = [reshape(squeeze(mean(expr.prob{1}, 1)), 1, []) ...
              reshape(squeeze(mean(expr.prob{1}, 2)), 1, []) ...
              reshape(squeeze(mean(expr.prob{1}, 3)), 1, [])];
    
        vo = reshape(squeeze(expr.prob{1}), 1, []);
    
        for cnt_trial_to_fit=1:length(ntriaToFit)
    
            cnt_trial = ntriaToFit(cnt_trial_to_fit);
            
            idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ; % 1-3
            idx_color(2)    = colorMap(inputTarget(2, cnt_trial))+3 ; % 4-6
            idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial))+6 ; % 7-9
            idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
            idx_color(1)    = colorMap(inputTarget(1, cnt_trial))+3 ;
            idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial))+6 ;
            idx_patternshape(1) = (idx_pattern(1)-7)*3 + idx_shape(1) ; % 1-9
            idx_patternshape(2) = (idx_pattern(2)-7)*3 + idx_shape(2) ; 
            assert(1<=idx_patternshape(1) & idx_patternshape(1)<=9 & 1<=idx_patternshape(2) & idx_patternshape(2)<=9);
            idx_patterncolor(1) = (idx_pattern(1)-7)*3 + (idx_color(1)-4)+10 ; % 10-18
            idx_patterncolor(2) = (idx_pattern(2)-7)*3 + (idx_color(2)-4)+10 ;
            assert(10<=idx_patterncolor(1) & idx_patterncolor(1)<=18 & 10<=idx_patterncolor(2) & idx_patterncolor(2)<=18);
            idx_shapecolor(1) = (idx_shape(1)-1)*3 + (idx_color(1)-4)+19 ;
            idx_shapecolor(2) = (idx_shape(2)-1)*3 + (idx_color(2)-4)+19 ; % 19-27
            assert(19<=idx_shapecolor(1) & idx_shapecolor(1)<=27 & 19<=idx_shapecolor(2) & idx_shapecolor(2)<=27);
    
    %         diff_vals(cnt_trial_to_fit,:) = [log(vf(idx_color(2))/vf(idx_color(1))) ...
    %                                   log(vc(idx_patternshape(2))/vc(idx_patternshape(1)))];
            
            options(cnt_trial_to_fit,:) = [pair_to_ind(idx_shape(1), idx_shape(2))...
                                           pair_to_ind(idx_color(1)-3, idx_color(2)-3)...
                                           pair_to_ind(idx_pattern(1)-6, idx_pattern(2)-6)...
                                           ];
            diff_vals(cnt_trial_to_fit,:) = [log(vf(idx_shape(2))/vf(idx_shape(1))) ...
                                      log(vf(idx_color(2))/vf(idx_color(1))) ...
                                      log(vc(idx_patterncolor(2))/vc(idx_patterncolor(1))) ...
                                      log(vc(idx_patternshape(2))/vc(idx_patternshape(1))) ...
                                      log(vc(idx_shapecolor(2))/vc(idx_shapecolor(1))) ...
                                      log(vo(inputTarget(2, cnt_trial))/vo(inputTarget(1, cnt_trial)))];
        end
    
        % G = findgroups(round(diff_vals(:,2), 4));
        % all_choice_probs{1}(end+1,:) = splitapply(@mean,choicetrials(ntriaToFit)-1,G)';
        % 
        % G = findgroups(round(diff_vals(:,4), 4));
        % all_choice_probs{2}(end+1,:) = splitapply(@mean,choicetrials(ntriaToFit)-1,G)';
        
    %     G = findgroups(round(diff_vals(:,3), 4));
    %     all_choice_probs{3}(end+1,:) = splitapply(@mean,choicetrials(ntriaToFit)-1,G)';
    
        all_diff_vals = [all_diff_vals; (diff_vals) ones(length(ntriaToFit),1)*cnt_sbj];
    %     all_option_pairs = [all_option_pairs; options ones(length(ntriaToFit),1)*cnt_sbj];
        all_choicetrials = [all_choicetrials; choicetrials(ntriaToFit)-1];
    end
    
    tbls{idx_block} = array2table([all_choicetrials all_diff_vals], ...
        'VariableNames', ["Choice", "S", "C", "PC", "PS", "SC", "O", "subject"]);
end

%%


for idx_block=1:4

    
    mdls{idx_block, 1} = fitglme(tbls{idx_block}, "Choice ~ C+PS+O + (C+PS+O|subject)", 'Distribution','binomial',...
                'FitMethod', 'Laplace', 'Verbose', 1, 'CheckHessian',true);
    mdls{idx_block, 2} = fitglme(tbls{idx_block}, "Choice ~ PC+O + (PC+O|subject)", 'Distribution','binomial',...
                'FitMethod', 'Laplace', 'Verbose', 1, 'CheckHessian',true);
    mdls{idx_block, 3} = fitglme(tbls{idx_block}, "Choice ~ SC+O + (SC+O|subject)", 'Distribution','binomial',...
                'FitMethod', 'Laplace', 'Verbose', 1, 'CheckHessian',true);

end


% tbl = array2table([all_choicetrials all_option_pairs], ...
%     'VariableNames', ["Choice", "S", "C", "P", "subject"]);
% 
% mdl{6} = fitglme(tbl, "Choice ~ S*C*P-S:C:P+(S*C*P-S:C:P|subject)", 'Distribution','binomial',...
%             'FitMethod', 'Laplace', 'Verbose', 1);

% cd ../files/
% save("choice_curve_models", "mdls")
% cd ../analysis/

%% 

figure

clrmats = colormap('turbo(6)');
clrmats = clrmats(2:4, :);

all_AICs = [];

for idx_block=1:4
    all_AICs(1,idx_block) = mdls{idx_block,1}.ModelCriterion.AIC;
    all_AICs(2,idx_block) = mdls{idx_block,2}.ModelCriterion.AIC;
    all_AICs(3,idx_block) = mdls{idx_block,3}.ModelCriterion.AIC;
end

for idx_mdl=1:3
   plot(all_AICs(idx_mdl,:), 'Color', clrmats(idx_mdl,:), 'LineWidth', 2, 'Marker', 'o');
   hold on
end

xlim([0.8, 4.2]);
ylim(ylim()+0.1*[-1,1].*range(ylim()))

legend(["F_{inf}+C_{inf}", "C_{noninf1}", "C_{noninf2}"], ...
    "Location", "best", 'Orientation', 'vertical');
xlabel('Trial block')
ylabel('AIC')

box off

%%

all_res = [];

for idx_block=1:4
    [B, ~] = randomEffects(mdls{idx_block});
    B = reshape(B, [4, 67]);
    all_res(idx_block,:,:) = B(2:end,:)+mdls{idx_block}.Coefficients.Estimate(2:end);
    all_coeffs(:,idx_block) = mdls{idx_block}.Coefficients.Estimate(2:end);
    all_ses(:,idx_block) = mdls{idx_block}.Coefficients.SE(2:end);
    all_ts(:,idx_block) = mdls{idx_block}.Coefficients.tStat(2:end);
    all_ps(:,idx_block) = mdls{idx_block}.Coefficients.pValue(2:end);
end

ttt = tiledlayout(1,3);

axes = [];
labels = ["F_{inf}", "C_{inf}", "O"];

cmap = colormap('lines(7)');
cmap = cmap([1,4,7],:);

for idx_var=1:3
    axes(idx_var,:) = nexttile;
    alpha = 0.1;
    plot(squeeze(all_res(:,idx_var,:)), '-', ...
        'Color', [cmap(idx_var,:), alpha], 'LineWidth', 1);
    hold on;
    xxx = repmat(1:4, [67,1])';
    yyy = squeeze(all_res(:,idx_var,:));
    scatter(xxx(:), yyy(:), 50, 'o', 'MarkerEdgeAlpha', alpha, ...
        'MarkerEdgeColor', cmap(idx_var,:), 'LineWidth', 1);
    xlim([0.5, 4.5]);
    ylim(ylim()+0.1*[-1,1].*range(ylim()))
    % if idx_var<3
    %     xticks([])
    % end
    title(labels(idx_var), 'fontweight', 'normal');
    xticks(1:4)

    errorbar(all_coeffs(idx_var,:), all_ses(idx_var,:), 'Color', ...
        cmap(idx_var,:), 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 10, ...
        'MarkerFaceColor', 'white');
    hold on;
    box off
end

% linkaxes(axes, 'y');
xlabel(ttt, 'Trial block', 'FontSize', 25)
ylabel(ttt, 'Coefficient', 'FontSize', 25)



%%

all_coeff_names = ["F_{inf}", "C_{inf}", "O"];
all_coeff_names = repmat(all_coeff_names, [1,4]);
pe_mdl_coeffs_tbl = table(all_coeff_names(:), ...
                          round(all_coeffs(:), 2), ...
                          round(all_ses(:), 2), ...
                          round(all_ts(:), 2), ...
                          round(all_ps(:), 3), ...
                          'VariableNames', ["Name", "Estimate", "SE", "t", "p-value"]);
table2latex(pe_mdl_coeffs_tbl, '../tables/choice_curve_coeff_tbl')


