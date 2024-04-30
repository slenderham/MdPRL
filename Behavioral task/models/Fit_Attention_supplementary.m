clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../utils")
% addpath("../utils/DERIVESTsuite/DERIVESTsuite/")

%% load subjects

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
% subjects1 = ["AA", "AB"];
subjects1 = lower(subjects1);
subjects1_inputs = "inputs/input_"+subjects1;
subjects1_prl = "SubjectData/PRL_"+subjects1;

subjects2 = [...
    "AA", "AB", "AC", "AD", "AE", "AG", ...
    "AH", "AI", "AJ", "AK", "AL", "AM", "AN", ...
    "AO", "AP", "AQ", "AR", "AS", "AT", "AU", "AV", ...
    "AW", "AX", "AY"] ;
% subjects2 = ["AA", "AB"] ;
subjects2_inputs = "inputs2/input_"+subjects2;
subjects2_prl = "SubjectData2/PRL_"+subjects2;

subjects_inputs = [subjects1_inputs subjects2_inputs];
subjects_prl = [subjects1_prl subjects2_prl];


%% load model specs

attn_ops = ["diff", "sum", "max"];
attn_times = ["C", "L", "CL"];

[attn_ops, attn_times] = meshgrid(attn_ops, attn_times);
attn_modes = ["const", "none"; attn_ops(:) attn_times(:)];

all_model_names = ["fMLchoiceLL_RL2onlyconjdecayattn", ...
                   "fMLchoiceLL_RL2conjdecayattn_onlycattn"];

all_model_Nparambasic = [3, 4];
all_model_Nalphas = [2, 2];
all_model_Nbetas = [1, 1];

bound_eps = 0;
temp_bound_eps = 0;
bias_bound = 5;
p_bias_bound = 5;
ch_temp_bound = 500;
p_ch_temp_bound = 500;
attn_temp_bound = 500;
p_attn_temp_bound = 500;

all_lbs = {...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps]};

all_ubs = {...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound], ...
    [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, attn_temp_bound]};

all_plbs = {...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps], ...
    [-p_bias_bound, temp_bound_eps, bound_eps, bound_eps, bound_eps, bound_eps, temp_bound_eps]};

all_pubs = {...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound], ...
    [p_bias_bound, p_ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps, p_attn_temp_bound]};

nrep = 40;
% nrep = 2;

op = optimset('Display', 'off');
%% optimize

poolobj = parpool('local', 32);
for m = 1:length(all_model_names)
    disp("=======================================================");
    disp(strcat("Fitting model ", all_model_names(m)));
    basic_params = cell(length(subjects_inputs), 1); % store the attention-less model's parameters for each model type
    for a = 1:length(attn_modes)
        tic
        disp("-------------------------------------------------------");
        disp(strcat("Fitting attn type ", attn_modes(a, 1), " ", attn_modes(a, 2)));
        parfor cnt_sbj = 1:length(subjects_inputs)
            disp(strcat("Fitting subject ", num2str(cnt_sbj)));
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

            minfval = 1000000;
            
            for cnt_rep = 1:nrep
                sesdata = struct();
                sesdata.input   = input ;
                sesdata.expr    = expr ;
                sesdata.results = results ;
                sesdata.NtrialsShort = expr.NtrialsShort ;
                sesdata.flagUnr = 1 ;

                sesdata.flag_couple = 0 ;
                sesdata.flag_updatesim = 0 ;

                sesdata.use_rpe = false;

                % load parameter numbers
                NparamBasic = all_model_Nparambasic(m);
                sesdata.Nalpha = all_model_Nalphas(m);

                if a==1
                    sesdata.Nbeta = 0;
                else
                    sesdata.Nbeta = all_model_Nbetas(m);
                end

                % load attn type (const, diff, sum, max) and attn
                % time(none, choice, learning, both)
                sesdata.attn_op = attn_modes(a,1);
                sesdata.attn_time = attn_modes(a,2);

                % load lb and ub, initialize
                plbs = all_plbs{m};
                pubs = all_pubs{m};
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar = plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                ipar(plbs>=0 & pubs>20) = exp(log(plbs(plbs>=0 & pubs>20)+1)+ ...
                    rand(size(plbs(plbs>=0 & pubs>20))).*(log(pubs(plbs>=0 & pubs>20))-log(plbs(plbs>=0 & pubs>20)+1)));
                if a>1 && cnt_rep==1 % initialize with the no-attn model's parameters for one trial
                    ipar(1:length(basic_params{cnt_sbj})) = basic_params{cnt_sbj};
                end
                lbs = all_lbs{m};
                ubs = all_ubs{m};
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);

                % load model likelihood func and optimize
                ll = str2func(all_model_names(m));
                ll = @(x)sum(ll(x, sesdata));
                [xpar, fval, exitflag, output] = bads(ll, ipar, lbs, ubs, plbs, pubs, [], op) ;

                if fval <= minfval
                    minfval = fval ;
                    fit_results{m, a, cnt_sbj}.params = xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                    fit_results{m, a, cnt_sbj}.fval = fval ;
                    fit_results{m, a, cnt_sbj}.iters = output.iterations;
                    fit_results{m, a, cnt_sbj}.exitflag = exitflag ;
                    if a==1
                        basic_params{cnt_sbj} = xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                    end
%                     fit_results{m, a, cnt_sbj}.hessian = hessian(ll, xpar) ;
%                     fit_results.lbs = lbs ;
%                     fit_results.ubs = ubs ;
                end
            end
        end
        toc
        cd ../files
        save RPL2Analysis_Attention_suppl
        cd ../models
    end
end

cd ../files
save RPL2Analysis_Attention_suppl
cd ../models
