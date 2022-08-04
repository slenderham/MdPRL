clc
clear
close all
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
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

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
    "fMLchoiceLL_RL2ftobjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn", ...
    "fMLchoiceLL_RL2conjdecayattn_onlyfattn", ...
    "fMLchoiceLL_RL2conjdecayattn_constrained"];

attns = load('../files/RPL2Analysis_Attention_merged_rep50.mat') ;
ntrials = 432;

clear all_sim_choices all_sim_rewards all_sim_corrects all_sim_lls all_sim_values all_sim_attns
nreps = 50;
for m = [1 5]
    disp("=======================================================");
    disp(strcat("Simulating model ", all_model_names(m)));
    for a = [1 3]
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_sbj = 1:length(subjects_inputs)
            for cnt_rep = 1:nreps
                inputname   = strcat("../PRLexp/inputs_all/", subjects_inputs(cnt_sbj) , ".mat") ;
                resultsname = strcat("../PRLexp/SubjectData_all/", subjects_prl(cnt_sbj) , ".mat") ;
    
                inputs_struct = load(inputname);
                results_struct = load(resultsname);

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
    
                sim_model = str2func(all_sim_model_names(m));
                
                [~, latents] = sim_model(best_pars, sesdata);

                all_sim_choices(m, a, cnt_sbj, cnt_rep, :) = latents.C;
                all_sim_rewards(m, a, cnt_sbj, cnt_rep, :) = latents.R;
                [~, idxMax] = max(expr.prob{1}(input.inputTarget)) ;
                all_sim_corrects(m, a, cnt_sbj, cnt_rep, :) = sim_choices==idxMax;
                all_sim_correct_behav(m, a, cnt_sbj, cnt_rep, :) = latents.C==results.choice';
                all_sim_lls(m, a, cnt_sbj, cnt_rep, :) = -logsigmoid(latents.logits'.*(sesdata.results.choice*2-3));
                all_sim_values{m, a, cnt_sbj, cnt_rep} = latents.V;
                all_sim_attns{m, a, cnt_sbj, cnt_rep} = latents.A;
            end
        end
    end
end

