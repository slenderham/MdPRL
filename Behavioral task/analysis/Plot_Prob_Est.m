clc
close all
clear

set(0,'defaultAxesFontSize',25)
%%

addpath("../files")
addpath("../models")
addpath("../utils")

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
subjects_prl = [subjects1_prl subjects2_prl];

ntrialPerf       = 33:432;
perfTH = 0.53;
idxSubject       = 1:length(subjects_inputs);
wSize  = 20 ;

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
% idxperf(39) = false;
subjects_inputs = subjects_inputs(idxperf);
subjects_prl = subjects_prl(idxperf);

%% make design matrix

probeTrialsAll = load(['../PRLexp/inputs_all/inputs/input_', 'aa' , '.mat']).expr.trialProbe;

for cnt_probe = 1:length(probeTrialsAll)
    pEstAll{cnt_probe}   = nan*ones(length(subjects_inputs),27) ;
    XAll{cnt_probe}      = [] ;
end

clear sesdata
clear SS
clear all_prob_ests
clear b1 b21 b22 b23 b3
for cnt_sbj = 1:length(subjects_inputs)
    inputname   = strcat('../PRLexp/inputs_all/', subjects_inputs(cnt_sbj) , '.mat') ;
    resultsname = strcat('../PRLexp/SubjectData_all/', subjects_prl(cnt_sbj) , '.mat') ;

    load(inputname)
    load(resultsname)

    expr.shapeMap     = repmat([1 2 3 ;
        1 2 3 ;
        1 2 3 ], 1,1,3) ;

    expr.colorMap     = repmat([1 1 1 ;
        2 2 2 ;
        3 3 3], 1,1,3) ;

    expr.patternMap(:,:,1)      = ones(3,3) ;
    expr.patternMap(:,:,2)      = 2*ones(3,3) ;
    expr.patternMap(:,:,3)      = 3*ones(3,3) ;

    for cnt_probe = 1:length(results.probEst)
        probEst         = results.probEst{cnt_probe} ;
        probEst         = probEst./(1+probEst) ;
        %         probEst = log(probEst);

        % regression coefficient
        X   = 4 ;
        ll1 = [X      1  1/X;
            1/X^2  1  X^2;
            X      1  1/X]' ;

        X   = 3 ;
        ll  = [X      1  1/X]' ;

        LLSh            = nan*ones(3,3,3) ;
        LLSh(1,:,:)     = ll1*ll(1) ;
        LLSh(2,:,:)     = ones(size(ll1))*ll(2) ;
        LLSh(3,:,:)     = ll1*ll(3) ;

        % features
        Regft1 = mean(expr.prob{1}, [2 3]);
        Regft1 = repmat(Regft1, [1 3 3]);
        Regft2 = mean(expr.prob{1}, [1 3]);
        Regft2 = repmat(Regft2, [3 1 3]);
        Regft3 = mean(expr.prob{1}, [1 2]);
        Regft3 = repmat(Regft3, [3 3 1]);
        % Regft3 is constant so omitted

        % conjunctions
        Regconj1 = mean(expr.prob{1}, 1);
        Regconj1 = repmat(Regconj1, [3 1 1]);
        %         Regconj1 = Regft1.*Regconj1 ./ (Regft1.*Regconj1+(1-Regft1).*(1-Regconj1));
        Regconj2 = mean(expr.prob{1}, 2);
        Regconj2 = repmat(Regconj2, [1 3 1]);
        %         Regconj2 = Regft2.*Regconj2 ./ (Regft2.*Regconj2+(1-Regft2).*(1-Regconj2));
        Regconj3 = mean(expr.prob{1}, 3);
        Regconj3 = repmat(Regconj3, [1 1 3]);
        %         Regconj3 = Regft3.*Regconj3 ./ (Regft3.*Regconj3+(1-Regft3).*(1-Regconj3));

        % objects
        LLRegobj = LLSh ;
        Regobj = LLRegobj./(1+LLRegobj) ;

        XTEMP = [Regft1(expr.playcombinations); Regft2(expr.playcombinations); ...
            Regconj1(expr.playcombinations); Regconj2(expr.playcombinations); ...
            Regconj3(expr.playcombinations); Regobj(expr.playcombinations); ...
            probEst(expr.playcombinations)]' ;
        XTEMP = round(XTEMP./0.05)*0.05 ;
        if XTEMP<0.05
            XTEMP = 0.05;
        elseif XTEMP>0.95
            XTEMP = 0.95;
        end
        XTEMP = log(XTEMP+1e-10) - log(1-XTEMP+1e-10);
        XTEMP(isnan(XTEMP)) = 0;

        %         XALL(cnt_probe, cnt_sbj, :, :) = XTEMP;

        X = XTEMP(:,end);
        Y1 = XTEMP(:, 1);
        Y21 = XTEMP(:,[1 3]);
        Y22 = XTEMP(:, 4);
        Y23 = XTEMP(:, 5);
        Y3 = XTEMP(:, 6);


        all_prob_ests(:, cnt_probe, cnt_sbj) = X;

    end
end

%% fit random effect model

p_vals = nan*ones(5,1);
for cnt_probe=1:5
    tbl1 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat(Y1, [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'Finf', 'Subject'});
    tbl21 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat(Y21(:,1), [sum(idxperf), 1]), ...
        repmat(Y21(:,2), [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'Finf', 'Cinf', 'Subject'});
    tbl22 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat(Y22, [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'Cnoninf1', 'Subject'});
    tbl23 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat(Y23, [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'Cnoninf2', 'Subject'});
    tbl3 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat(Y3, [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'O', 'Subject'});
    tbl4 = table(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), [], 1), ...
        repmat((Y21(:,1)), [sum(idxperf), 1]), ...
        repmat((Y21(:,2)), [sum(idxperf), 1]), ...
        repmat((Y3), [sum(idxperf), 1]), ...
        repelem(1:sum(idxperf), 27)', ...
        'VariableNames', {'Y', 'Finf', 'Cinf', 'O', 'Subject'});
    
%     disp("------------")
%     disp([1, cnt_probe])
    mdl1 = fitlme(tbl1, 'Y~Finf+(Finf|Subject)','CovariancePattern','FullCholesky','FitMethod','ML','CheckHessian',true);
%     disp("------------")
%     disp([2, cnt_probe])
    mdl21 = fitlme(tbl21, 'Y~Finf+Cinf+(Finf+Cinf|Subject)','CovariancePattern','FullCholesky','FitMethod','ML','CheckHessian',true);
%     disp("------------")
%     disp([3, cnt_probe])
    mdl22 = fitlme(tbl22, 'Y~Cnoninf1+(Cnoninf1|Subject)','CovariancePattern','FullCholesky','FitMethod','ML','CheckHessian',true);
%     disp("------------")
%     disp([4, cnt_probe])
    mdl23 = fitlme(tbl23, 'Y~Cnoninf2+(Cnoninf2|Subject)','CovariancePattern','FullCholesky','FitMethod','ML','CheckHessian',true);
%     disp("------------")
%     disp([5, cnt_probe])
    mdl3 = fitlme(tbl3, 'Y~O+(O|Subject)','CovariancePattern','FullCholesky','FitMethod','ML','CheckHessian',true);
%     disp("------------")
%     disp([6, cnt_probe])
    mdl4 = fitlme(tbl4, 'Y~Finf+Cinf+O+(1|Subject)','CovariancePattern','Diagonal','FitMethod','REML','CheckHessian',true);
%     disp("------------")

    mdls{1, cnt_probe} = mdl1;
    mdls{2, cnt_probe} = mdl21;
    mdls{3, cnt_probe} = mdl22;
    mdls{4, cnt_probe} = mdl23;
    mdls{5, cnt_probe} = mdl3;
    mdls{6, cnt_probe} = mdl4;

    RsqS(1, cnt_probe) = mdl1.Rsquared.Adjusted;
    RsqS(2, cnt_probe) = mdl21.Rsquared.Adjusted;
    RsqS(3, cnt_probe) = mdl22.Rsquared.Adjusted;
    RsqS(4, cnt_probe) = mdl23.Rsquared.Adjusted;
    RsqS(5, cnt_probe) = mdl3.Rsquared.Adjusted;
%     RsqS(6, cnt_probe) = mdl4.Rsquared.Adjusted;

%     betas(cnt_probe, 1, :) = [mdl21.Coefficients.Estimate(2), mdl21.Coefficients.SE(2)]';
    betas(cnt_probe, 1, :) = [mdl4.Coefficients.Estimate(2), mdl4.Coefficients.SE(2)]';
%     betas(cnt_probe, 2, :) = [mdl21.Coefficients.Estimate(3), mdl21.Coefficients.SE(3)]';
    betas(cnt_probe, 2, :) = [mdl4.Coefficients.Estimate(3), mdl4.Coefficients.SE(3)]';
    betas(cnt_probe, 3, :) = [mdl22.Coefficients.Estimate(2), mdl22.Coefficients.SE(2)]';
    betas(cnt_probe, 4, :) = [mdl23.Coefficients.Estimate(2), mdl23.Coefficients.SE(2)]';
%     betas(cnt_probe, 5, :) = [mdl3.Coefficients.Estimate(2), mdl3.Coefficients.SE(2)]';
    betas(cnt_probe, 5, :) = [mdl4.Coefficients.Estimate(4), mdl4.Coefficients.SE(4)]';
%     betas(cnt_probe, 6, :) = [mdl3.Coefficients.Estimate(2), mdl5.Coefficients.SE(2)]';
end



%% population level anova

%          nominal(repelem(1:sum(idxperf), 3^3))},...


parfor cnt_probe = 1:length(results.probEst)
    [p,tbl,stats,~] = anovan(reshape(squeeze(all_prob_ests(:,cnt_probe,:)), 1, []), ...
        {nominal(repmat(expr.shapeMap(expr.playcombinations), [1, sum(idxperf)])), ...
        nominal(repmat(expr.colorMap(expr.playcombinations), [1, sum(idxperf)])), ...
        nominal(repmat(expr.patternMap(expr.playcombinations), [1, sum(idxperf)])), ...
        nominal(repelem(1:sum(idxperf), 3^3))},...
        "model",3,"varnames",["shape", "color", "pattern", "subject"],'display','on','random',4);
    anova_mdls{cnt_probe}.tbl = tbl;
    anova_mdls{cnt_probe}.stats = stats;
    anova_mdls{cnt_probe}.p = p;
    anova_ps(cnt_probe, :) = p;
end

for cnt_probe = 1:length(results.probEst)
    tbl = anova_mdls{cnt_probe}.tbl;
    eta2(cnt_probe, :) = ([tbl{2:end-2,2}])./([tbl{end-1,2}]+[tbl{2:end-2,2}]);
end
eta2_fixed = eta2(:, [1 2 3 5 6 8 11]);

% cd ../files/
% save("prob_est_models", "mdls", "anova_mdls")
% cd ../analysis/

%% plot different levels of different strategy

omega_steps = 1000;

omega_feat = [ones(1, omega_steps) (1-(0:omega_steps)./omega_steps)]*1;
omega_conj = [((0:omega_steps-1)./omega_steps) ones(1, omega_steps+1)]*1;

for i=1:length(omega_conj)
    expr.prob{1} = round(expr.prob{1}/0.05)*0.05;
    expr.prob{1}(expr.prob{1}>0.95) = 0.95;
    expr.prob{1}(expr.prob{1}<0.05) = 0.05;

    Regft1 = mean(expr.prob{1}, [2 3]);
    Regft1 = repmat(Regft1, [1 3 3]);
    %     Regft1 = log(Regft1+1e-10) - log(1-Regft1+1e-10);
    Regft2 = mean(expr.prob{1}, [1 3]);
    Regft2 = repmat(Regft2, [3 1 3]);
    %     Regft2 = log(Regft2+1e-10) - log(1-Regft2+1e-10);
    Regft3 = mean(expr.prob{1}, [1 2]);
    Regft3 = repmat(Regft3, [3 3 1]);
    %     Regft3 = log(Regft3+1e-10) - log(1-Regft3+1e-10);

    Regconj1 = mean(expr.prob{1}, 1);
    Regconj1 = repmat(Regconj1, [3 1 1]);
    %     Regconj1 = log(Regconj1+1e-10) - log(1-Regconj1+1e-10);
    Regconj2 = mean(expr.prob{1}, 2);
    Regconj2 = repmat(Regconj2, [1 3 1]);
    %     Regconj2 = log(Regconj2+1e-10) - log(1-Regconj2+1e-10);
    Regconj3 = mean(expr.prob{1}, 3);
    Regconj3 = repmat(Regconj3, [1 1 3]);
    %     Regconj3 = log(Regconj3+1e-10) - log(1-Regconj3+1e-10);

    %     approx = 1./(1+exp(-omega*Regft1 - (1-omega)*Regconj1));
    approx = Regft1.^(omega_feat(i)).*Regconj1.^(omega_conj(i))./...
        (Regft1.^(omega_feat(i)).*Regconj1.^(omega_conj(i))+...
        (1-Regft1).^(omega_feat(i)).*(1-Regconj1).^(omega_conj(i)));
    strats_err(1, i) = -mean(expr.prob{1}.*log(approx)+(1-expr.prob{1}).*log(1-approx), 'all');
    %     approx = 1./(1+exp(-omega*Regft2 - (1-omega)*Regconj2));
    approx = Regft2.^(omega_feat(i)).*Regconj2.^(omega_conj(i))./...
        (Regft2.^(omega_feat(i)).*Regconj2.^(omega_conj(i))+...
        (1-Regft2).^(omega_feat(i)).*(1-Regconj2).^(omega_conj(i)));
    strats_err(2, i) = -mean(expr.prob{1}.*log(approx)+(1-expr.prob{1}).*log(1-approx), 'all');
    %     approx = 1./(1+exp(-omega*Regft3 - (1-omega)*Regconj3));
    approx = Regft3.^(omega_feat(i)).*Regconj3.^(omega_conj(i))./...
        (Regft3.^(omega_feat(i)).*Regconj3.^(omega_conj(i))+...
        (1-Regft3).^(omega_feat(i)).*(1-Regconj3).^(omega_conj(i)));
    strats_err(3, i) = -mean(expr.prob{1}.*log(approx)+(1-expr.prob{1}).*log(1-approx), 'all');
end

%%

figure
clrmats = colormap('turbo(6)');
clrmats = clrmats(1:5, :);
for i=1:5
    plot(RsqS(i,:), 'Color', clrmats(i,:), 'LineWidth', 1, 'Marker', 'o');hold on;
end
ylim([0.0, 0.4])
legend(["F_{inf}", "F_{inf}+C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"], ...
    "Location", "eastoutside", 'Orientation', 'vertical');
xticks(1:5)
xlabel('Value estimation bout')
ylabel('Adj. R^2')
xlim([0.75, 5.25])
set(gca, 'box','off')

figure
clrmats = colormap('lines(7)');
eb = errorbar(betas(:,[1 2 5],1), betas(:,[1 2 5],2), '-o', 'LineWidth', 1);
eb(1).Color = clrmats(1,:);
eb(2).Color = clrmats(4,:);
eb(3).Color = clrmats(7,:);
xlim([0.75, 5.25])
ylim([-0.05, 0.62])
legend(["F_{inf}", "C_{inf}", "O"], "Location", "eastoutside", 'Orientation', 'vertical');
xlabel('Value estimation bout')
ylabel('Regression weights')
set(gca, 'box','off')

% figure
% prob_est_rmses = (XALL(:,:,:,4)-reshape(expr.prob{1}(expr.playcombinations), [1 1 27])).^2;
% plot_shared_errorbar(nanmean(prob_est_rmses, [2 3]), nanstd(prob_est_rmses, [], [2 3])/sqrt(size(XALL, 2)));

figure;
plot(eta2_fixed(:,[2 1 3 5 6 4 7]), '-o', 'LineWidth', 1); hold on;
xlim([0.75, 5.25])
ylim([-0.01, 0.6])
xticks(1:5)
xlabel('Value estimation bout')
ylabel('\eta_p^2', 'FontSize', 25)
legend(["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"], ...
    "Location", "eastoutside", 'Orientation', 'vertical');
set(gca, 'box','off')

%%
figure;
plot(strats_err', 'LineWidth', 2);
yline(-mean(log(1/2),'all'), "--")
legend(["F_{inf}+C_{inf}", "F_{noninf1}+C_{noninf1}", "F_{noninf2}+C_{noninf2}", "Chance"]);
xticks([0, length(omega_feat)/2, length(omega_feat)]);
xticklabels([])
xlim([1, length(omega_feat)])
% yticks(0.02:0.01:0.06)
ylim([0.59, 0.7])
xlabel('\leftarrow Feat.         Mixed           Conj. \rightarrow')
% ylabel('E$_O[\ KL(p_{r}(O)||\overline {p_{r}}(O))\ ]$', 'interpreter', 'latex')
ylabel('Approximation error')
box off


%% fit and plot sum of squared explained by each dimension
[p,tbl,stats,terms] = anovan(expr.prob{1}(:), ...
    {nominal(expr.shapeMap(expr.playcombinations)), ...
    nominal(expr.colorMap(expr.playcombinations)), ...
    nominal(expr.patternMap(expr.playcombinations))},...
    "model",2, "varnames",["shape", "color", "pattern"],'display','off');
clrmats = brighten(colormap('lines(7)'), 0);
figure;
b = bar([tbl{[3 2 4 6 5 7 8],2}]'/tbl{9,2});
b.FaceColor = 'flat';
b.CData = clrmats;
xticklabels(["F_{inf}", "F_{noninf1}", "F_{noninf2}", "C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"]);
ylabel('Prop. variance explained')
xlim([0.4, 7.6])
set(gca, "fontsize", 22)
box off

%% plot mean and exact reward probs by feature
clrmats = colormap('lines(7)');
ft_values = [];
for i=1:3
    dims = 1:3;
    ft_values(end+1,:) = squeeze(mean(expr.prob{1}, dims(dims~=i)));
end
b = bar(ft_values);
box off
hold on;
xticklabels({'F_{inf}', 'F_{noninf1}', 'F_{noninf2}'});
ylabel('Reward probabilites')
bx = [];
for i=1:3
    b(i).FaceColor = 'flat';
    b(i).FaceAlpha = 0.3;
    b(i).CData = clrmats(1:3,:);
    bx(:,i) = b(i).XEndPoints;
end

for i=1:3
    true_probs = permute(expr.prob{1}, mod([1:3]+i+1,3)+1);
    true_probs = reshape(true_probs, 3, []);
    dims = 1:3;
    errorbar(bx(i,:),squeeze(mean(expr.prob{1}, ...
        dims(dims~=i))),squeeze(std(expr.prob{1}, [], dims(dims~=i))),...
        'linestyle', 'none', 'Color',clrmats(i,:), 'LineWidth',1);
    scatter(zeros(9,1)+bx(i,:), true_probs'+(2*rand(9,3)-1)*0.01, 50, clrmats(i,:), ...
        'LineWidth', 1, 'MarkerEdgeAlpha',.5);
end

%% plot mean and exact reward probs by conjunction
% conj_values = [];
% for i=1:3
%     conj_values(end+1,:) = reshape(squeeze(mean(expr.prob{1}, i)), [], 1);
% end
% b = bar(conj_values);
% hold on;
% xticklabels({'C_{inf}', 'C_{noninf1}', 'C_{noninf2}'});
% ylabel('Reward Probabilites')
% bx = [];
% for i=1:length(b)
%     b(i).FaceColor = 'flat';
%     b(i).FaceAlpha = 0.3;
%     b(i).CData = clrmats(4:6,:);
%     bx(:,i) = b(i).XEndPoints;
% end
% 
% dim_orders = [1 2 3; 2 1 3; 3 1 2];
% 
% for i=1:3
%     true_probs = permute(expr.prob{1}, dim_orders(i,:));
%     true_probs = reshape(true_probs, [], 9);
%     scatter(zeros(3,1)+bx(i,:), true_probs+(2*rand(3,9)-1)*0.01, 50, clrmats(i+3,:), 'LineWidth', 1);
% end

%% make table for manuscript


all_model_names = ["F_{inf}", "F_{inf}+C_{inf}", "C_{noninf1}", "C_{noninf2}", "O"];
probEst_trials = string([86, 173, 259, 346, 432]);
all_coeff_names = ["F_{inf}", "C_{inf}", "O"];
clear pe_results

for cnt_probe=1:length(results.probEst)
    for cnt_mdl=1:5
        if cnt_mdl==1
            pe_results.Trial(cnt_mdl, cnt_probe) = probEst_trials(cnt_probe);
        else
            pe_results.Trial(cnt_mdl, cnt_probe) = "";
        end
        pe_results.ModelType{cnt_mdl, cnt_probe} = all_model_names(cnt_mdl);
        pe_results.AIC(cnt_mdl, cnt_probe) = mdls{cnt_mdl, cnt_probe}.ModelCriterion.AIC;
        pe_results.BIC(cnt_mdl, cnt_probe) = mdls{cnt_mdl, cnt_probe}.ModelCriterion.BIC;
        pe_results.LL(cnt_mdl, cnt_probe) = mdls{cnt_mdl, cnt_probe}.ModelCriterion.LogLikelihood;
    end    
    pe_results.name{1, cnt_probe} = all_coeff_names(1);
    pe_results.betas(1, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.Estimate(2), 2);
    pe_results.ts(1, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.tStat(2), 3);
    pe_results.ses(1, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.SE(2), 2);
    pe_results.ps(1, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.pValue(2), 3);
    pe_results.name{2, cnt_probe} = all_coeff_names(2);
    pe_results.betas(2, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.Estimate(3), 2);
    pe_results.ts(2, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.tStat(3), 3);
    pe_results.ses(2, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.SE(3), 2);
    pe_results.ps(2, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.pValue(3), 3);
    pe_results.name{3, cnt_probe} = all_coeff_names(3);
    pe_results.betas(3, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.Estimate(4), 2);
    pe_results.ts(3, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.tStat(4), 2);
    pe_results.ses(3, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.SE(4), 3);
    pe_results.ps(3, cnt_probe) = round(mdls{6, cnt_probe}.Coefficients.pValue(4), 3);
end

%%
pe_mdl_comparison_tbl = array2table(["$"+[pe_results.ModelType{1:5}]'+"$", ...
                              "\makecell{-LL="+round(-pe_results.LL, 2)+"\\"+ ...
                              "AIC="+round(pe_results.AIC, 2)+"\\"+ ...
                              "BIC="+round(pe_results.BIC, 2)+"}"],...
                              'VariableNames', ["Model", "Est. bout 1", "Est. bout 2", ...
                                                "Est. bout 3", "Est. bout 4", "Est. bout 5"]);
table2latex(pe_mdl_comparison_tbl, '../tables/pe_mdl_comparison_tbl')

pe_mdl_coeffs_tbl = table("$"+[pe_results.name{:}]'+"$", ...
                          pe_results.betas(:), ...
                          pe_results.ses(:), ...
                          pe_results.ps(:), ...
                          'VariableNames', ["Coefficient", "Estimate", "SE", "p-value"]);
table2latex(pe_mdl_coeffs_tbl, '../tables/pe_mdl_coeffs_tbl')

