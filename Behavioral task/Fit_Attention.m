clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("PRLexp/inputs/")
addpath("PRLexp/SubjectData/")

%%

subjects = {...
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', ...
    'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', ...
    'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', ...
    'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', ...
    'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', ...
    'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'CC', 'DD', ...
    'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', ...
    'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', ...
    'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ'} ;

% subjects = {'AA'};

attn_modes = {'diff', 'sum', 'max'}; %'constant' TODO: constant
len_i_1 = length(attn_modes);
len_i_2 = length(attn_modes);

nrep        = 5 ;
randstate   = clock ;

% op          = optimset('MaxFunEvals', 4000, 'MaxIter', 4000);
op          = optimset('Display', 'off');
poolobj = parpool('local', 16);
%%
parfor cnt_sbj = 1:length(subjects)
    inputname   = ['./PRLexp/inputs/input_', lower(subjects{cnt_sbj}) , '.mat'] ;
    resultsname = ['./PRLexp/SubjectData/PRL_', lower(subjects{cnt_sbj}) , '.mat'] ;

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

    fvalminRL2_ft_attn = ones(4)*10000;
    fvalminRL2_ftobj_attn = ones(4)*10000;
    fvalminRL2_ftconj_attn = ones(4)*10000;

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

                %% RL2 Feature decay
                sesdata.flag_couple = 0 ;
                sesdata.flag_updatesim = 0 ;
                sesdata.flagSepAttn = 1;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                NparamBasic = 3 ;
                if sesdata.flagUnr==1
                    sesdata.Nalpha = 2 ;
                else
                    sesdata.Nalpha = 1 ;
                end
                sesdata.Nbeta = 2;
                ipar= [rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta)];
                ll = @(x)fMLchoicefit_RL2ftdecayattn(x, sesdata);
                lbs = [-20, 0,  0, 0, 0, 0,  0];
                ubs = [ 20, 20, 1, 1, 1, 20, 20];
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                if fval <= fvalminRL2_ft_attn(i1, i2)
                    fvalminRL2_ft_attn(i1, i2) = fval ;
                    mlparRL2_ft_decay_attn{i1, i2, cnt_sbj}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2_ft_decay_attn{i1, i2, cnt_sbj}(100) = fval ;
                    mlparRL2_ft_decay_attn{i1, i2, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
                    mlparRL2_ft_decay_attn{i1, i2, cnt_sbj}(102) = output.iterations;
                    mlparRL2_ft_decay_attn{i1, i2, cnt_sbj}(103) = exitflag ;
                end

                %% RL2 Feature+Obj decay
                sesdata.flag_couple = 0 ;
                NparamBasic = 4 ;
                sesdata.flatSepAttn = 1;
                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                sesdata.Nbeta = 2;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                ipar= [rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta)];
                ll = @(x)fMLchoicefit_RL2ftobjdecayattn(x, sesdata);
                lbs = [-20,  0,  0, 0, 0, 0, 0, 0,  0,  0];
                ubs = [ 20, 20, 20, 1, 1, 1, 1, 1, 20, 20];
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [],[],[],[], lbs, ubs, [], op) ;
                if fval <= fvalminRL2_ftobj_attn(i1, i2)
                    fvalminRL2_ftobj_attn(i1, i2) = fval ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}(100) = fval ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}(102) = output.iterations;
                    mlparRL2ftobj_decay_attn{i1, i2, cnt_sbj}(103) = exitflag ;
                end

                %% RL2 conjunction decay
                sesdata.flag_couple = 0 ;
                NparamBasic = 4 ;
                sesdata.flatSepAttn = 1;
                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                sesdata.Nbeta = 4;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                ipar= [rand(1,NparamBasic+sesdata.Nalpha+sesdata.Nbeta)];
                ll = @(x)fMLchoicefit_RL2conjdecayattn(x, sesdata);
                lbs = [-20,  0,  0, 0, 0, 0, 0, 0,  0,  0,  0,  0];
                ubs = [ 20, 20, 20, 1, 1, 1, 1, 1, 20, 20, 20, 20];
                [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
                if fval <= fvalminRL2_ftconj_attn(i1, i2)
                    fvalminRL2_ftconj_attn(i1, i2) = fval ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)= (xpar(1:NparamBasic+sesdata.Nalpha+sesdata.Nbeta)) ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}(100) = fval ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}(102) = output.iterations;
                    mlparRL2conj_decay_attn{i1, i2, cnt_sbj}(103) = exitflag ;
                end

                %%
            end
        end
    end
end

cd ./files
save RPL2Analysis_Attention
cd ../