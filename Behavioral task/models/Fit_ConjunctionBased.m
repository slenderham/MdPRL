clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../utils")
addpath("../utils/bads")
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

bound_eps = 0;
temp_bound_eps = 0;
bias_bound = 5;
ch_temp_bound = 500;

lbs = [-bias_bound, bound_eps, bound_eps, bound_eps, bound_eps, bound_eps];
ubs = [bias_bound, ch_temp_bound, 1-bound_eps, 1-bound_eps, 1-bound_eps, 1-bound_eps];
plbs = lbs;
pubs = ubs;

nrep = 40;

op = optimset('Display', 'off');

%%
poolobj = parpool('local', 16);
parfor cnt_sbj = 1:length(subjects_inputs)
    disp(['Subject: ', num2str(cnt_sbj)])
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
    
    for cntD = 1:3

        minfval = 1000000;

        for cnt_rep  = 1:nrep
            %% RL2 conjunction coupled
            % sesdata.flag_updatesim = 0 ;
            % sesdata.flag_couple = 1 ;
            % NparamBasic = 3 ;
            % if sesdata.flagUnr==1
            %     sesdata.Nalpha = 4 ;
            % else
            %     sesdata.Nalpha = 2 ;
            % end
            % ipar= 0.1*[rand(1,NparamBasic+sesdata.Nalpha)]  ;
            % [xpar fval exitflag output] = fminsearch(@fMLchoicefit_RL2conj, ipar, op, sesdata) ;
            % if fval <= fvalminRL2_couple
            %     xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha])=1./(1+exp(-(xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha]))./sesdata.sig) ) ;
            %     fvalminRL2_couple = fval ;
            %     mlparRL2conj_couple{cntD, cnt_sbj}(1:NparamBasic+sesdata.Nalpha)= (xpar(1:NparamBasic+sesdata.Nalpha)) ;
            %     mlparRL2conj_couple{cntD, cnt_sbj}(100) = fval ;
            %     mlparRL2conj_couple{cntD, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
            %     mlparRL2conj_couple{cntD, cnt_sbj}(102) = output.iterations;
            %     mlparRL2conj_couple{cntD, cnt_sbj}(103) = exitflag ;
            % end

            %% RL2 conjunction uncoupled
            % sesdata.flag_updatesim = 0 ;
            % sesdata.flag_couple = 0 ;
            % NparamBasic = 3 ;
            % if sesdata.flagUnr==1
            %     sesdata.Nalpha = 4 ;
            % else
            %     sesdata.Nalpha = 2 ;
            % end
            % ipar= 0.1*[rand(1,NparamBasic+sesdata.Nalpha)]  ;
            % [xpar fval exitflag output] = fminsearch(@fMLchoicefit_RL2conj, ipar, op, sesdata) ;
            % if fval <= fvalminRL2_uncouple
            %     xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha])=1./(1+exp(-(xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha]))./sesdata.sig) ) ;
            %     fvalminRL2_uncouple = fval ;
            %     mlparRL2conj_uncouple{cntD, cnt_sbj}(1:NparamBasic+sesdata.Nalpha)= (xpar(1:NparamBasic+sesdata.Nalpha)) ;
            %     mlparRL2conj_uncouple{cntD, cnt_sbj}(100) = fval ;
            %     mlparRL2conj_uncouple{cntD, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
            %     mlparRL2conj_uncouple{cntD, cnt_sbj}(102) = output.iterations;
            %     mlparRL2conj_uncouple{cntD, cnt_sbj}(103) = exitflag ;
            % end

            %% RL2 conjunction decay
            sesdata = struct();
            sesdata.input   = input ;
            sesdata.expr    = expr ;
            sesdata.results = results ;
            sesdata.NtrialsShort = expr.NtrialsShort ;
            sesdata.flagUnr = 1 ;
            sesdata.cntD = cntD ;
            sesdata.flag_couple = 0 ;
            sesdata.flag_updatesim = 0 ;
        
            NparamBasic = 4 ;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 2 ;
            else
                sesdata.Nalpha = 1 ;
            end

            ipar = plbs+rand(1,NparamBasic+sesdata.Nalpha).*(pubs-plbs);
            ipar(plbs>=0 & pubs>20) = exp(log(plbs(plbs>=0 & pubs>20)+1)+ ...
                    rand(size(plbs(plbs>=0 & pubs>20))).*(log(pubs(plbs>=0 & pubs>20))-log(plbs(plbs>=0 & pubs>20)+1)));
            ll = @(x)sum(fMLchoiceLL_RL2conjdecay(x, sesdata));

            [xpar, fval, exitflag, output] = bads(ll, ipar, lbs, ubs, plbs, pubs, [], op) ;
            if fval <= minfval
                minfval = fval ;
                fit_results{cntD, cnt_sbj}.params = (xpar(1:NparamBasic+sesdata.Nalpha)) ;
                fit_results{cntD, cnt_sbj}.fval = fval ;
                fit_results{cntD, cnt_sbj}.iters = output.iterations;
                fit_results{cntD, cnt_sbj}.exitflag = exitflag ;
            end
        end
    end
end

cd ../files
save RPL2Analysis_Baseline_ConjunctionBased
cd ../models