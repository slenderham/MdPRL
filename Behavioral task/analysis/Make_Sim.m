clc
clear
close all
rng('shuffle')
randstate = clock ;
addpath("../PRLexp/inputs/")
addpath("../PRLexp/SubjectData/")
addpath("../utils")
addpath("../../PRLexpv3_5v2/")

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

attn_modes = {'const', 'diff', 'sum', 'max'};
len_i_1 = length(attn_modes);
len_i_2 = length(attn_modes);

all_model_names = ["fMLchoiceSim_RL2ftdecayattn", ...
                    "fMLchoiceSim_RL2ftobjdecayattn", ...
                    "fMLchoiceSim_RL2conjdecayattn", ...
                    "fMLchoiceSim_RL2conjdecayattn_constrained"];

nSamples = 100 ;
trialLength = 432;
nrep = 5;

op = optimset('Display', 'none');
poolobj = parpool('local', 16);


for cnt_samp = 1:nSamples
    fGenerateInputIndividual(num2str(cnt_samp,'%03.f'));
end

disp('Started Simulation')

for cnt_samp  = 1:nSamples
    % draw random subject stimuli, randomly permute
    draw_sub_data = randsample(length(subjects));
    rand_order = randperm(trialLength);

    inputname   = ['../PRLexp/inputs_sim/input_', lower(subjects{draw_sub_data}) , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData/PRL_', lower(subjects{draw_sub_data}) , '.mat'] ;

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


parfor cnt_sbj = 1:length(subjects)
    inputname   = ['../PRLexp/inputs/input_', lower(subjects{cnt_sbj}) , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData/PRL_', lower(subjects{cnt_sbj}) , '.mat'] ;

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

    for cnt_samp  = 1:nSamples
        disp(['Subject: ', num2str(cnt_sbj),', Sample: ', num2str(cnt_samp)])

        for i1 = 1:len_i_1
            for i2 = 1:len_i_2
                sesdata = struct();
                sesdata.sig     = 1 ;
                sesdata.input   = input ;
                sesdata.expr    = expr ;
                sesdata.results = results ;
                sesdata.NtrialsShort = expr.NtrialsShort ;
                sesdata.flagUnr = 1 ;

                % RL2 Feature decay
                sesdata.flag_couple = 0 ;
                sesdata.flag_updatesim = 0 ;
                sesdata.flagSepAttn = 1;

                sesdata.NparamBasic = 3 ;
                if sesdata.flagUnr==1
                    sesdata.Nalpha = 2 ;
                else
                    sesdata.Nalpha = 1 ;
                end
                sesdata.Nbeta = 2;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};
                ipars = [2*rand(1)-1 ...
                    exp(log(1)+rand(1)*(log(50)-log(1))) ...
                    exp(log(0.005)+rand(1)*(log(0.005)-log(0.05))) ...
                    exp(log(0.05)+rand(1, 2)*(log(0.05)-log(0.5))) ...
                    exp(log(1)+rand(1, 2)*(log(50)-log(1)))];

                [Cs, Rs, Vs, As] = fMLchoiceSim_RL2ftdecayattn(ipars, sesdata);
                all_ft_Cs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Cs;
                all_ft_Rs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Rs;

                % RL2 Feature+Obj decay
                sesdata.NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                sesdata.Nbeta = 2;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};

                ipars = [2*rand(1)-1 ...
                    exp(log(1)+rand(1, 2)*(log(50)-log(1))) ...
                    exp(log(0.005)+rand(1)*(log(0.005)-log(0.05))) ...
                    exp(log(0.05)+rand(1, 4)*(log(0.05)-log(0.5))) ...
                    exp(log(1)+rand(1, 2)*(log(50)-log(1)))];

                [Cs, Rs, Vs, As] = fMLchoiceSim_RL2ftobjdecayattn(ipars, sesdata);
                all_ftobj_Cs{cnt_sbj, cnt_samp, i1, i2, 1} = Cs;
                all_ftobj_Rs{cnt_sbj, cnt_samp, i1, i2, 1} = Rs;

                % RL2 conjunction decay
                sesdata.NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                sesdata.Nbeta = 4;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};

                ipars = [2*rand(1)-1 ...
                    exp(log(1)+rand(1, 2)*(log(50)-log(1)))...
                    exp(log(0.005)+rand(1)*(log(0.005)-log(0.05))) ...
                    exp(log(0.05)+rand(1, 4)*(log(0.05)-log(0.5))) ...
                    exp(log(1)+rand(1, 4)*(log(50)-log(1)))];

                [Cs, Rs, Vs, As] = fMLchoiceSim_RL2conjdecayattn(ipars, sesdata);
                all_ftconj_Cs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Cs;
                all_ftconj_Rs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Rs;


                % RL2 conjunction decay constrained attn
                sesdata.NparamBasic = 4 ;

                if sesdata.flagUnr==1
                    sesdata.Nalpha = 4 ;
                else
                    sesdata.Nalpha = 2 ;
                end
                sesdata.Nbeta = 4;
                sesdata.attn_mode_choice = attn_modes{i1};
                sesdata.attn_mode_learn = attn_modes{i2};

                ipars = [2*rand(1)-1 ...
                    exp(log(1)+rand(1, 2)*(log(50)-log(1)))...
                    exp(log(0.005)+rand(1)*(log(0.005)-log(0.05))) ...
                    exp(log(0.05)+rand(1, 4)*(log(0.05)-log(0.5))) ...
                    exp(log(1)+rand(1, 4)*(log(50)-log(1)))];

                [Cs, Rs, Vs, As] = fMLchoiceSim_RL2conjdecayattn_constrained(ipars, sesdata);
                all_ftconj_constr_Cs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Cs;
                all_ftconj_constr_Rs{cnt_sbj, cnt_samp, i1, i2, 1}(1:trialLength) = Rs;

            end
        end
    end
end

all_Cs = cat(5, all_ft_Cs, all_ftobj_Cs, all_ftconj_Cs, all_ftconj_constr_Cs);
all_Rs = cat(5, all_ft_Rs, all_ftobj_Rs, all_ftconj_Rs, all_ftconj_constr_Rs);

perfMeans = zeros(len_i_1, len_i_2, 4);

ntrialPerf = 33:432;
for cnt_sbj = 1:length(subjects)
    inputname   = ['../PRLexp/inputs/input_', lower(subjects{cnt_sbj}) , '.mat'] ;
    resultsname = ['../PRLexp/SubjectData/PRL_', lower(subjects{cnt_sbj}) , '.mat'] ;
    load(inputname)
    load(resultsname)
    [~, idxMax] = max(expr.prob{1}(input.inputTarget)) ;

    disp(['Subject ' num2str(cnt_sbj)])
    for cnt_samp = 1:nSamples
        for i1=1:len_i_1
            for i2=1:len_i_2
                choice_better(1,:) =  idxMax==all_ft_Cs{cnt_sbj, cnt_samp, i1, i2};
                choice_better(2,:) =  idxMax==all_ftobj_Cs{cnt_sbj, cnt_samp, i1, i2};
                choice_better(3,:) =  idxMax==all_ftconj_Cs{cnt_sbj, cnt_samp, i1, i2};
                choice_better(4,:) = idxMax==all_ftconj_constr_Cs{cnt_sbj, cnt_samp, i1, i2};

                perfMeans(i1, i2, :) = perfMeans(i1, i2, :) + reshape(nanmean(choice_better, 2)/length(subjects)/nSamples, [1, 1, 4]) ;
            end
        end
    end
end

