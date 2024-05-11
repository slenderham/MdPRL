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
ntrialsToFit = 200;

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

all_diff_vals = [];
all_rts = [];
all_option_pairs = [];

all_rt_grouped = {[], []};

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
    rts = results.responsetime(end-(ntrialsToFit-1):end);

    % shape, color, pattern
    vf = [squeeze(mean(expr.prob{1}, [1 3])) ...
          squeeze(mean(expr.prob{1}, [2 3]))' ...
          squeeze(mean(expr.prob{1}, [1 2]))']; 

    % patternshape patterncolor shapecolor
    vc = [reshape(squeeze(mean(expr.prob{1}, 1)), 1, []) ...
          reshape(squeeze(mean(expr.prob{1}, 2)), 1, []) ...
          reshape(squeeze(mean(expr.prob{1}, 3)), 1, [])];

    vo = reshape(squeeze(expr.prob{1}), 1, []);

    for cnt_trial=ntrials-(ntrialsToFit-1):ntrials        
        idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ; % 1-3
        idx_color(2)    = 3+colorMap(inputTarget(2, cnt_trial)) ; % 4-6
        idx_pattern(2)  = 6+patternMap(inputTarget(2, cnt_trial)) ; % 7-9
        idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
        idx_color(1)    = 3+colorMap(inputTarget(1, cnt_trial)) ;
        idx_pattern(1)  = 6+patternMap(inputTarget(1, cnt_trial)) ;
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
        
        options(cnt_trial-(ntrials-(ntrialsToFit-1))+1,:) = [pair_to_ind(idx_shape(1), idx_shape(2))...
                                       pair_to_ind(idx_color(1)-3, idx_color(2)-3)...
                                       pair_to_ind(idx_pattern(1)-6, idx_pattern(2)-6)...
                                       ];
        diff_vals(cnt_trial-(ntrials-(ntrialsToFit-1))+1,:) = [log(vf(idx_shape(2))/vf(idx_shape(1))) ...
                                  log(vf(idx_color(2))/vf(idx_color(1))) ...
                                  log(vc(idx_patterncolor(2))/vc(idx_patterncolor(1))) ...
                                  log(vc(idx_patternshape(2))/vc(idx_patternshape(1))) ...
                                  log(vc(idx_shapecolor(2))/vc(idx_shapecolor(1))) ...
                                  log(vo(inputTarget(2, cnt_trial))/vo(inputTarget(1, cnt_trial)))];
    end

    
    G = findgroups(diff_vals(:,2));
    all_rt_grouped{1}(end+1,:) = splitapply(@mean,rts,G)';
    
    G = findgroups(diff_vals(:,4));
    all_rt_grouped{2}(end+1,:) = splitapply(@mean,rts,G)';
    
%     G = findgroups(round(diff_vals(:,3), 4));
%     all_rt_grouped{3}(end+1,:) = splitapply(@mean,choicetrials(ntriaToFit)-1,G)';

    use_idx = 1:ntrialsToFit;
    for idx=[86, 173, 259, 346]+1
        use_idx = use_idx(use_idx~=idx);
    end
    
    all_diff_vals = [all_diff_vals; diff_vals(use_idx,:) ones(length(use_idx),1)*cnt_sbj];
%     all_option_pairs = [all_option_pairs; options ones(ntrialsToFit,1)*cnt_sbj];
    all_rts = [all_rts; log(rts(use_idx))];
end

all_diff_vals = all_diff_vals(all_rts>log(0.1),:);
all_rts = all_rts(all_rts>log(0.1),:);

%%
tbl = array2table([all_rts abs(all_diff_vals)], ...
    'VariableNames', ["RT", "S", "C", "PC", "PS", "SC", "O", "subject"]);

mdl{1} = fitlme(tbl, "RT ~ C+(1|subject)", 'Verbose', 0, 'CovariancePattern', 'Diagonal', 'CheckHessian', true);

mdl{2} = fitlme(tbl, "RT ~ C+PS+(1|subject)", 'Verbose', 0, 'CovariancePattern', 'Diagonal', 'CheckHessian', true);

mdl{3} = fitlme(tbl, "RT ~ S+PC+(1|subject)", 'Verbose', 0, 'CovariancePattern', 'Diagonal', 'CheckHessian', true);

mdl{4} = fitlme(tbl, "RT ~ SC+(1|subject)", 'Verbose', 0, 'CovariancePattern', 'Diagonal', 'CheckHessian', true);

mdl{5} = fitlme(tbl, "RT ~ O+(1|subject)", 'Verbose', 0, 'CovariancePattern', 'Diagonal', 'CheckHessian', true);

% tbl = array2table([all_rts all_option_pairs], ...
%     'VariableNames', ["RT", "S", "C", "P", "subject"]);
% 
% mdl{6} = fitglme(tbl, "RT ~ S*C*P-S:C:P+(S*C*P-S:C:P|subject)", 'Distribution','binomial',...
%             'FitMethod', 'Laplace', 'Verbose', 1);

% cd ../files/
% save("choice_curve_models", "mdls")
% cd ../analysis/

%% 

figure

cmap = colormap('lines(6)');
cmap = cmap([1, 4],:);

log_odds_x = -1.:0.01:1.;
% ll = plot(log_odds_x, 1./(1+exp(-mdl{2}.Coefficients.Estimate(2:end)*log_odds_x))', 'LineWidth',1);
% for i=1:length(ll)
%     ll(i).Color = cmap(i,:);
% end


hold on;
C = unique(all_diff_vals(:,2));
errorbar(C, mean(all_rt_grouped{1}), std(all_rt_grouped{1})/sqrt(length(idxperf)), "-o", 'Color', cmap(1,:));
C = unique(all_diff_vals(:,4));
errorbar(C, mean(all_rt_grouped{2}), std(all_rt_grouped{2})/sqrt(length(idxperf)), "-o", 'Color', cmap(2,:));

legend(ll, {'F_{inf}', 'C_{inf}'}, 'Location', 'southeast')

xlabel('Log Odds of Reward')
ylabel('Choice Probability')
xlim([-1., 1.])
ylim([0.13, 0.87])

axes('Position',[.55 .17 .2 .2])
box on
bb = bar(mdl{2}.Coefficients.Estimate(2:end)); hold on
bb.FaceColor = 'flat';
bb.CData = cmap(1:2,:);
errorbar(mdl{2}.Coefficients.Estimate(2:end), mdl{2}.Coefficients.SE(2:end), 'k', "LineStyle","none");
xlim([0.5, 2.5])
ylim([0, 2.3])
xticks(1:2)
xticklabels([])
ylabel("Slopes", 'FontSize',18)
text(1-0.22,mdl{2}.Coefficients.Estimate(2)+mdl{2}.Coefficients.SE(2)+0.1,'****','Color', cmap(1,:),'FontSize',18)
text(2-0.1,mdl{2}.Coefficients.Estimate(3)+mdl{2}.Coefficients.SE(3)+0.1,'**','Color', cmap(2,:),'FontSize',18)

% %%
% for i=1:5
%     rsqs(i) = mdl{i}.ModelCriterion.Deviance;
% end
% bar(rsqs)
