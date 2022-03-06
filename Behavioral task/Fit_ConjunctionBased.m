
clc
clear
close all
rng('shuffle')
randstate = clock ;

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

nrep        = 2 ; 
randstate   = clock ;

op          = optimset;
sesdata.flagUnr = 1 ;

%%
for cnt_sbj = 1:length(subjects)
    inputname   = ['./PRLexp/inputs/input_', subjects{cnt_sbj} , '.mat'] ;
    resultsname = ['./PRLexp/SubjectData/PRL_', subjects{cnt_sbj} , '.mat'] ;
    
    load(inputname)
    load(resultsname)
    
    expr.shapeMap = repmat([1 2 3 ;
                    1 2 3 ;
                    1 2 3 ], 1,1,3) ;

    expr.colorMap = repmat([1 1 1 ;
                    2 2 2 ;
                    3 3 3], 1,1,3) ;
                
    expr.patternMap(:,:,1) = ones(3,3) ;
    expr.patternMap(:,:,2) = 2*ones(3,3) ;
    expr.patternMap(:,:,3) = 3*ones(3,3) ;
    
    %%
    
    sesdata.sig     = 0.2 ;
    sesdata.input   = input ;
    sesdata.expr    = expr ;
    sesdata.results = results ;
    sesdata.NtrialsShort = expr.NtrialsShort ;

    for cntD = 1:3
        fvalminRL2_couple       = length(sesdata.results.reward) ;
        fvalminRL2_uncouple     = length(sesdata.results.reward) ;
        fvalminRL2_decay        = length(sesdata.results.reward) ;
        sesdata.cntD            = cntD ;
        
        for cnt_rep  = 1:nrep
            disp(['----------------------------------------------'])
            disp(['Subject: ', num2str(cnt_sbj),', Repeat: ', num2str(cnt_rep)])

            %% RL2 conjunction coupled
%             sesdata.flag_updatesim = 0 ;
%             sesdata.flag_couple = 1 ;
%             NparamBasic = 3 ;
%             if sesdata.flagUnr==1
%                 sesdata.Nalpha = 4 ;
%             else
%                 sesdata.Nalpha = 2 ;
%             end
%             ipar= 0.1*[rand(1,NparamBasic+sesdata.Nalpha)]  ;
%             [xpar fval exitflag output] = fminsearch(@fMLchoicefit_RL2conj, ipar, op, sesdata) ;
%             if fval <= fvalminRL2_couple
%                 xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha])=1./(1+exp(-(xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha]))./sesdata.sig) ) ;
%                 fvalminRL2_couple = fval ;
%                 mlparRL2conj_couple{cntD, cnt_sbj}(1:NparamBasic+sesdata.Nalpha)= (xpar(1:NparamBasic+sesdata.Nalpha)) ;
%                 mlparRL2conj_couple{cntD, cnt_sbj}(100) = fval ;
%                 mlparRL2conj_couple{cntD, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
%                 mlparRL2conj_couple{cntD, cnt_sbj}(102) = output.iterations;
%                 mlparRL2conj_couple{cntD, cnt_sbj}(103) = exitflag ;
%             end

            %% RL2 conjunction uncoupled
%             sesdata.flag_updatesim = 0 ;
%             sesdata.flag_couple = 0 ;
%             NparamBasic = 3 ;
%             if sesdata.flagUnr==1
%                 sesdata.Nalpha = 4 ;
%             else
%                 sesdata.Nalpha = 2 ;
%             end
%             ipar= 0.1*[rand(1,NparamBasic+sesdata.Nalpha)]  ;
%             [xpar fval exitflag output] = fminsearch(@fMLchoicefit_RL2conj, ipar, op, sesdata) ;
%             if fval <= fvalminRL2_uncouple
%                 xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha])=1./(1+exp(-(xpar([NparamBasic+1:NparamBasic+sesdata.Nalpha]))./sesdata.sig) ) ;
%                 fvalminRL2_uncouple = fval ;
%                 mlparRL2conj_uncouple{cntD, cnt_sbj}(1:NparamBasic+sesdata.Nalpha)= (xpar(1:NparamBasic+sesdata.Nalpha)) ;
%                 mlparRL2conj_uncouple{cntD, cnt_sbj}(100) = fval ;
%                 mlparRL2conj_uncouple{cntD, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
%                 mlparRL2conj_uncouple{cntD, cnt_sbj}(102) = output.iterations;
%                 mlparRL2conj_uncouple{cntD, cnt_sbj}(103) = exitflag ;
%             end

            %% RL2 conjunction decay
            sesdata.flag_couple = 0 ;
            NparamBasic = 4 ;
            if sesdata.flagUnr==1
                sesdata.Nalpha = 4 ;
            else
                sesdata.Nalpha = 2 ;
            end
            ipar= rand(1,NparamBasic+sesdata.Nalpha);
            ll = @(x)fMLchoicefit_RL2conjdecay(x, sesdata);
            lbs = [-20, 0,  0,  0, 0, 0, 0, 0, 0];
            ubs = [ 20, 20, 20, 1, 1, 1, 1, 1, 1];
            [xpar, fval, exitflag, output] = fmincon(ll, ipar, [], [], [], [], lbs, ubs, [], op) ;
            if fval <= fvalminRL2_decay
                fvalminRL2_decay = fval ;
                mlparRL2conj_decay{cntD, cnt_sbj}(1:NparamBasic+sesdata.Nalpha)= (xpar(1:NparamBasic+sesdata.Nalpha)) ;
                mlparRL2conj_decay{cntD, cnt_sbj}(100) = fval ;
                mlparRL2conj_decay{cntD, cnt_sbj}(101) = fval./length(sesdata.results.reward) ;
                mlparRL2conj_decay{cntD, cnt_sbj}(102) = output.iterations;
                mlparRL2conj_decay{cntD, cnt_sbj}(103) = exitflag ;
            end
        end
    end
end

cd ./files
save RPL2Analysisv3_5_ConjunctionBased
cd ../
