clc
close all
clear

set(0,'defaultAxesFontSize',18)
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

all_var_names = ["C", "S", "P", "PS", "PC", "SC"]';
all_var_names = all_var_names';
all_var_names = all_var_names(:);
all_var_names = all_var_names + ["xR", "xNR"];
all_var_names = all_var_names';
all_var_names = all_var_names(:);

%%
ca_mdls = load("../files/credit_assignment_models.mat");
pe_mdls = load("../files/prob_est_models.mat");
    
for cnt_probe=1:5
    curr_ca_mdl = ca_mdls.mdls{cnt_probe};
    [ca_re_flat, ca_names, ~] = randomEffects(curr_ca_mdl);
    for i=1:length(all_var_names)
        ca_re(cnt_probe,i,:) = ca_re_flat(ca_names.Name==all_var_names(i));
    end

    curr_pe_mdl = pe_mdls.mdls{2, cnt_probe};
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,1,:) = pe_re_flat(pe_names.Name=="Finf");
    pe_re(cnt_probe,2,:) = pe_re_flat(pe_names.Name=="Cinf");
    curr_pe_mdl = pe_mdls.mdls{3, cnt_probe};
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,3,:) = pe_re_flat(pe_names.Name=="Cnoninf1");
    curr_pe_mdl = pe_mdls.mdls{4, cnt_probe};
    [pe_re_flat, pe_names, ~] = randomEffects(curr_pe_mdl);
    pe_re(cnt_probe,4,:) = pe_re_flat(pe_names.Name=="Cnoninf2");

    subplot(2,5,cnt_probe)
    scatter(squeeze(ca_re(cnt_probe,1,:)-ca_re(cnt_probe,2,:))', squeeze(pe_re(cnt_probe,1,:))')
    disp(corr(squeeze(ca_re(cnt_probe,1,:)-ca_re(cnt_probe,2,:)), squeeze(pe_re(cnt_probe,1,:)),"type","Spearman"))
    xlim([-1 1])
    ylim([-1.5, 1.5])
    lsline
    subplot(2,5,5+cnt_probe)
    scatter(squeeze(ca_re(cnt_probe,7,:)-ca_re(cnt_probe,8,:))', squeeze(pe_re(cnt_probe,2,:))')
    disp(corr(squeeze(ca_re(cnt_probe,7,:)-ca_re(cnt_probe,8,:)), squeeze(pe_re(cnt_probe,2,:)),'type','Spearman'))
    xlim([-0.4, 0.4])
    ylim([-0.6, 0.6])
    lsline
end

