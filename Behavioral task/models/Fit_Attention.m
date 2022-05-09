clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs_all/")
addpath("../PRLexp/SubjectData_all/")
addpath("../utils")
addpath("../utils/DERIVESTsuite/DERIVESTsuite/")

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

% 
% subjects = [subjects]

% subjects = {'AA'};

attn_modes = {'const', 'diff', 'sum', 'max'};
len_i_1 = length(attn_modes);
len_i_2 = length(attn_modes);

all_model_names = ["fMLchoiceLL_RL2ftdecayattn", ...
                   "fMLchoiceLL_RL2ftobjdecayattn", ...
                   "fMLchoiceLL_RL2conjdecayattn", ...
                   "fMLchoiceLL_RL2conjdecayattn_onlyfattn", ...
                   "fMLchoiceLL_RL2conjdecayattn_constrained"];

nrep        = 10;

op          = optimset('Display', 'off', 'MaxIter', 4000);
poolobj = parpool('local', 16);

parfor cnt_sbj = 1:length(subjects_inputs)
    inputname   = ['../PRLexp/inputs_all/', subjects_inputs{cnt_sbj} , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData_all/', subjects_prl{cnt_sbj} , '.mat'] ;

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

    fvalminRL2_ft_attn = ones(len_i_1, len_i_2)*10000;
    fvalminRL2_ftobj_attn = ones(len_i_1, len_i_2)*10000;
    fvalminRL2_ftconj_attn = ones(len_i_1, len_i_2)*10000;
    fvalminRL2_ftconj_attn_onlyfattn = ones(len_i_1, len_i_2)*10000;
    fvalminRL2_ftconj_attn_constr = ones(len_i_1, len_i_2)*10000;

    for cnt_rep  = 1:nrep
%         disp(['----------------------------------------------'])
        disp(['Subject: ', num2str(cnt_sbj),', Repeat: ', num2str(cnt_rep)])

        for i1 = 1:len_i_1
            for i2 = 1:len_i_2
%                 disp(['Attention For Choice: ', attn_modes{i1},', Learning: ', attn_modes{i2}])
                sesdata = struct();
                sesdata.sig     = 1 ;
                sesdata.input   = input ;
                sesdata.expr    = expr ;
                sesdata.results = results ;
                sesdata.NtrialsShort = expr.NtrialsShort ;
                sesdata.flagUnr = 1 ;
                sesdata.flagSepAttn = 1;

                %% RL2 Feature decay
                sesdata.flag_couple = 0 ;
                sesdata.flag_updatesim = 0 ;
                
                NparamBasic = 3 ;
                if sesdata.flagUnr==1
                    sesdata.Nalpha = 2 ;
                else
                    sesdata.Nalpha = 1 ;
                end

                if i1==1 && i2==1
                    sesdata.Nbeta = 0;
                elseif i1==1 || i2==1
                    sesdata.Nbeta = 1;
                else
                    sesdata.Nbeta = 2;
                end

                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                
                plbs = [-1,  0, 0, 0, 0,  0,  0];
                pubs = [ 1, 10, 1, 1, 1, 10, 10];
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar= plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                
                ll = @(x)sum(fMLchoiceLL_RL2ftdecayattn(x, sesdata));
                
                lbs = [-50,  0, 0, 0, 0,  0,  0];
                ubs = [ 50, 50, 1, 1, 1, 50, 50];
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                
                if fval <= fvalminRL2_ft_attn(i1, i2)
                    fvalminRL2_ft_attn(i1, i2) = fval ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.params(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta) = (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.fval = fval ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.avgfval = fval./length(sesdata.results.reward) ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.iters = output.iterations;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.exitflag = exitflag ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.hessian = hessian(ll, xpar) ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.lbs = lbs ;
                    mlparRL2ft_decay_attn{i1, i2, cnt_sbj}.ubs = ubs ;
                end

                %% RL2 Feature+Obj decay
                NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                
                if i1==1 && i2==1
                    sesdata.Nbeta = 0;
                elseif i1==1 || i2==1
                    sesdata.Nbeta = 1;
                else
                    sesdata.Nbeta = 2;
                end

                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                
                plbs = [-1,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                pubs = [ 1, 10, 1, 1, 1, 1, 1, 1, 10, 10];
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar= plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                
                ll = @(x)sum(fMLchoiceLL_RL2ftobjdecayattn(x, sesdata));
                
                lbs = [-50,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                ubs = [ 50, 50, 1, 1, 1, 1, 1, 1, 50, 50];
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [],[],[],[], lbs, ubs, [], op) ;
                
                if fval <= fvalminRL2_ftobj_attn(i1, i2)
                    fvalminRL2_ftobj_attn(i1, i2) = fval ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.params(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.fval = fval ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.avgfval = fval./length(sesdata.results.reward) ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.iters = output.iterations;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.exitflag = exitflag ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.hessian = hessian(ll, xpar) ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.lbs = lbs ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}.ubs = ubs ;
                end

                %% RL2 conjunction decay
                NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                
                if i1==1 && i2==1
                    sesdata.Nbeta = 0;
                elseif i1==1 || i2==1
                    sesdata.Nbeta = 1;
                else
                    sesdata.Nbeta = 2;
                end
                
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                
                plbs = [-1,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                pubs = [ 1, 10, 1, 1, 1, 1, 1, 1, 10, 10];
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar= plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                
                ll = @(x)sum(fMLchoiceLL_RL2conjdecayattn(x, sesdata));
                
                lbs = [-50,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                ubs = [ 50, 50, 1, 1, 1, 1, 1, 1, 50, 50];
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                
                if fval <= fvalminRL2_ftconj_attn(i1, i2)
                    fvalminRL2_ftconj_attn(i1, i2) = fval ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.params(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.fval = fval ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.avgfval = fval./length(sesdata.results.reward) ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.iters = output.iterations;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.exitflag = exitflag ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.hessian = hessian(ll, xpar) ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.lbs = lbs ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}.ubs = ubs ;
                end

                %% RL2 conjunction decay with only feature attention
                NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                
                if i1==1 && i2==1
                    sesdata.Nbeta = 0;
                elseif i1==1 || i2==1
                    sesdata.Nbeta = 1;
                else
                    sesdata.Nbeta = 2;
                end

                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                
                plbs = [-1,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                pubs = [ 1, 10, 1, 1, 1, 1, 1, 1, 10, 10];
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar= plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                
                ll = @(x)sum(fMLchoiceLL_RL2conjdecayattn_onlyfattn(x, sesdata));
                
                lbs = [-50,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                ubs = [ 50, 50, 1, 1, 1, 1, 1, 1, 50, 50];
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                
                if fval <= fvalminRL2_ftconj_attn_onlyfattn(i1, i2)
                    fvalminRL2_ftconj_attn_onlyfattn(i1, i2) = fval ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.params(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.fval = fval ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.avgfval = fval./length(sesdata.results.reward) ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.iters = output.iterations;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.exitflag = exitflag;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.hessian = hessian(ll, xpar) ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.lbs = lbs ;
                    mlparRL2conj_decay_attn_onlyfattn{i1, i2, cnt_sbj}.ubs = ubs ;
                end

                %% RL2 conjunction decay constrained attn
                NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                
                if i1==1 && i2==1
                    sesdata.Nbeta = 0;
                elseif i1==1 || i2==1
                    sesdata.Nbeta = 1;
                else
                    sesdata.Nbeta = 2;
                end
                
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                
                plbs = [-1,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                pubs = [ 1, 10, 1, 1, 1, 1, 1, 1, 10, 10];
                plbs = plbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                pubs = pubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ipar= plbs+rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta).*(pubs-plbs);
                
                ll = @(x)sum(fMLchoiceLL_RL2conjdecayattn_constrained(x, sesdata));
                
                lbs = [-50,  0, 0, 0, 0, 0, 0, 0,  0,  0];
                ubs = [ 50, 50, 1, 1, 1, 1, 1, 1, 50, 50];
                lbs = lbs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                ubs = ubs(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta);
                
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                
                if fval <= fvalminRL2_ftconj_attn_constr(i1, i2)
                    fvalminRL2_ftconj_attn_constr(i1, i2) = fval ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.params(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.fval = fval ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.avgfval = fval./length(sesdata.results.reward) ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.iters = output.iterations;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.exitflag = exitflag;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.hessian = hessian(ll, xpar) ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.lbs = lbs ;
                    mlparRL2conj_decay_attn_constr{i1, i2, cnt_sbj}.ubs = ubs ;
                end
            end
        end
    end
end

cd ../files
save RPL2Analysis_Attention
cd ../models