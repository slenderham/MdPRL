clc
close all
clear
addpath("../files")
addpath("../models")
addpath("../utils")
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
idxperf = find(idxperf);
subjects_inputs = subjects_inputs(idxperf);
subjects_prl = subjects_prl(idxperf);

all_var_names = ["C", "S", "P", "PS", "PC", "SC"]';
all_var_names = all_var_names';
all_var_names = all_var_names(:);
all_var_names = all_var_names + ["_R", "_NR"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);

all_anova_var_names = ["shape", "color", "pattern", "subject"];

%%
ca_mdls = load("../files/credit_assignment_models_diagonal_cov.mat");
pe_mdls = load("../files/prob_est_models.mat");
anova_mdls = pe_mdls.anova_mdls;

% get anova coefficients
for cnt_probe = 1:5
    for i=1:3
        anova_fe{1}(1, cnt_probe, i) = anova_mdls{cnt_probe}.stats.coeffs( ...
            convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="shape="+i);
        anova_fe{1}(2, cnt_probe, i) = anova_mdls{cnt_probe}.stats.coeffs(...
            convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="color="+i);
        anova_fe{1}(3, cnt_probe, i) = anova_mdls{cnt_probe}.stats.coeffs(...
            convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="pattern="+i);
        anova_re{1}(1, cnt_probe, i,:) = anova_mdls{cnt_probe}.stats.coeffs( ...
            startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), "shape="+i+" * subject="));
        anova_re{1}(2, cnt_probe, i,:) = anova_mdls{cnt_probe}.stats.coeffs(...
            startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), "color="+i+" * subject="));
        anova_re{1}(3, cnt_probe, i,:) = anova_mdls{cnt_probe}.stats.coeffs(...
            startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), "pattern="+i+" * subject="));
        for j=1:3
            anova_fe{2}(1, cnt_probe,(i-1)*3+j) = anova_mdls{cnt_probe}.stats.coeffs(...
                convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="color="+i+" * pattern="+j);
            anova_fe{2}(2, cnt_probe,(i-1)*3+j) = anova_mdls{cnt_probe}.stats.coeffs(...
                convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="shape="+i+" * pattern="+j);
            anova_fe{2}(3, cnt_probe,(i-1)*3+j) = anova_mdls{cnt_probe}.stats.coeffs(...
                convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="shape="+i+" * color="+j);
            anova_re{2}(1, cnt_probe,(i-1)*3+j,:) = anova_mdls{cnt_probe}.stats.coeffs( ...
                startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), ...
                "color="+i+" * pattern="+j+" * subject="));
            anova_re{2}(2, cnt_probe,(i-1)*3+j,:) = anova_mdls{cnt_probe}.stats.coeffs(...
                startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), ...
                "shape="+i+" * pattern="+j+" * subject="));
            anova_re{2}(3, cnt_probe,(i-1)*3+j,:) = anova_mdls{cnt_probe}.stats.coeffs(...
                startsWith(convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames), ...
                "shape="+i+" * color="+j+" * subject="));
            for k=1:3
                anova_fe{3}(1, cnt_probe,(i-1)*9+(j-1)*3+k) = anova_mdls{cnt_probe}.stats.coeffs(...
                    convertCharsToStrings(anova_mdls{cnt_probe}.stats.coeffnames)=="shape="+i+" * color="+j+" * pattern="+k);
            end
        end
    end
end

anova_ind{1} = anova_re{1}+anova_fe{1};
anova_ind{2} = anova_re{2}+anova_fe{2};

%%  

for cnt_probe=1:5
    curr_ca_mdl = ca_mdls.mdls{cnt_probe};
    ca_fe(cnt_probe,:) = curr_ca_mdl.Coefficients.Estimate(2:end);
    [ca_re_flat, ca_names, ~] = randomEffects(curr_ca_mdl);
    for i=1:length(all_var_names)
        ca_re(cnt_probe,i,:) = ca_re_flat(ca_names.Name==all_var_names(i));
    end
    ca_ind(cnt_probe,:,:) = ca_fe(cnt_probe,:)+ca_re(cnt_probe,:,:);

    curr_pe_mdl = pe_mdls.mdls{2, cnt_probe};
    pe_fe(cnt_probe,[1 2]) = curr_pe_mdl.Coefficients.Estimate(2:end);
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,1,:) = pe_re_flat(pe_names.Name=="Finf");
    pe_re(cnt_probe,2,:) = pe_re_flat(pe_names.Name=="Cinf");
    pe_ind(cnt_probe,1,:) = pe_fe(cnt_probe,1)+pe_re(cnt_probe,1,:);
    pe_ind(cnt_probe,2,:) = pe_fe(cnt_probe,2)+pe_re(cnt_probe,2,:);
    
    curr_pe_mdl = pe_mdls.mdls{3, cnt_probe};
    pe_fe(cnt_probe,3) = curr_pe_mdl.Coefficients.Estimate(2);
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,3,:) = pe_re_flat(pe_names.Name=="Cnoninf1");
    pe_ind(cnt_probe,3,:) = pe_fe(cnt_probe,2)+pe_re(cnt_probe,3,:);
    
    curr_pe_mdl = pe_mdls.mdls{4, cnt_probe};
    pe_fe(cnt_probe,4) = curr_pe_mdl.Coefficients.Estimate(2);
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,4,:) = pe_re_flat(pe_names.Name=="Cnoninf2");
    pe_ind(cnt_probe,4,:) = pe_fe(cnt_probe,2)+pe_re(cnt_probe,4,:);
    
    figure(1)
    subplot(2,5,cnt_probe)
    scatter(squeeze(ca_re(cnt_probe,1,:)-ca_re(cnt_probe,2,:))', squeeze(pe_re(cnt_probe,1,:))')
    [r, p] = corr(squeeze(ca_re(cnt_probe,1,:)-ca_re(cnt_probe,2,:)), squeeze(pe_re(cnt_probe,1,:)), "type", "Spearman");
    disp([r, p]);
    xlim([-1 1]);
    ylim([-1.5, 1.5]);
    lsline
    subplot(2,5,5+cnt_probe)
    scatter(squeeze(ca_re(cnt_probe,7,:)-ca_re(cnt_probe,8,:))', squeeze(pe_re(cnt_probe,2,:))')
    [r, p] = corr(squeeze(ca_re(cnt_probe,7,:)-ca_re(cnt_probe,8,:)), squeeze(pe_re(cnt_probe,2,:)), "type", "Spearman");
    disp([r, p]);
    xlim([-0.4, 0.4]);
    ylim([-0.6, 0.6]);
    lsline    
% 
    figure(2)
    subplot(2,5,cnt_probe)
    imagesc(corr(squeeze(std(anova_ind{1}([2 1 3],cnt_probe,:,:), [], 3))', squeeze(ca_re(cnt_probe,1:2:6,:)-ca_re(cnt_probe,2:2:6,:))', 'type', 'spearman'))
    pbaspect([1 1 1])
    caxis([-1 1])
    colormap bluewhitered
    subplot(2,5,5+cnt_probe)
    imagesc(corr(squeeze(std(anova_ind{2}([2 1 3],cnt_probe,:,:), [], 3))', squeeze(ca_re(cnt_probe,7:2:end,:)-ca_re(cnt_probe,8:2:end,:))', 'type', 'spearman'))
    pbaspect([1 1 1])
    caxis([-1 1])
    colormap bluewhitered
end




%%
sim_vars = load('../files/simulated_vars');
all_probes = [0, 86, 173, 259, 346, 432];
for cnt_probe=1:5
    for i=1:length(idxperf)
        attn_ws(cnt_probe, i, :) = squeeze(mean(sim_vars.all_attns{5,3,idxperf(i)}(2,:,all_probes(cnt_probe)+1:all_probes(cnt_probe+1)), 3));
    end
end