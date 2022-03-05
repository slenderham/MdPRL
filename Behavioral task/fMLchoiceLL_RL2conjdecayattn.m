function [loglikehood, V, A] = fMLchoiceLL_RL2conjdecayattn(xpar, sesdata)
%
% DESCRIPTION: fits data to RL(2)obj model using ML method
%
% INPUT:
% sesdata structure which includes input, experiment and behavioral data
%
% OUTPUT:
% fitted parametres

loglikehood = 0 ;
NparamBasic = 4 ;
% xpar(2:3)= abs(xpar(2:3)) ;

BiasL = xpar(1) ;
magF  = xpar(2) ;
magC  = xpar(3) ;

% xpar([NparamBasic:NparamBasic+sesdata.Nalpha])=1./(1+exp(-(xpar([NparamBasic:NparamBasic+sesdata.Nalpha]))./sesdata.sig) ) ;
decay = xpar(4) ;
alpha_rewColor      = xpar([NparamBasic+1]) ;
alpha_rewShape      = xpar([NparamBasic+1]) ;
alpha_rewPattern    = xpar([NparamBasic+1]) ;
alpha_rew           = xpar([NparamBasic+2]) ;
NparamWithLR = NparamBasic+2;
if sesdata.flagUnr==1
    alpha_unrColor      = xpar([NparamBasic+3]) ;
    alpha_unrShape      = xpar([NparamBasic+3]) ;
    alpha_unrPattern    = xpar([NparamBasic+3]) ;
    alpha_unr           = xpar([NparamBasic+4]) ;
    NparamWithLR = NparamBasic+4;
else
    alpha_unrColor      = alpha_rewColor ;
    alpha_unrShape      = alpha_rewShape ;
    alpha_unrPattern    = alpha_rewPattern ;
    alpha_unr           = alpha_rew ;
end

beta_attn_choice_feat = xpar([NparamWithLR+1]);
beta_attn_choice_conj = xpar([NparamWithLR+2]);
if sesdata.flagSepAttn==1
    beta_attn_learn_feat = xpar([NparamWithLR+3]);
    beta_attn_learn_conj = xpar([NparamWithLR+4]);
else
    beta_attn_learn_feat = beta_attn_choice_feat;
    beta_attn_learn_conj = beta_attn_choice_conj;
end


shapeMap        = sesdata.expr.shapeMap ;
colorMap        = sesdata.expr.colorMap ;
patternMap      = sesdata.expr.patternMap ;
inputTarget     = sesdata.input.inputTarget ;
correcttrials   = sesdata.results.reward ;
choicetrials    = sesdata.results.choice ;
flag_couple     = sesdata.flag_couple ;
flag_updatesim  = sesdata.flag_updatesim ;
ntrials         = length(choicetrials) ;
inputRewards    = sesdata.input.inputReward ;

vf              = (0.5*ones(9,1)) ;
vc              = (0.5*ones(27,1)) ;

for cnt_trial=1:ntrials

    correct = correcttrials(cnt_trial) ;
    choice = choicetrials(cnt_trial) ;
    correctunCh = inputRewards(3-choice, cnt_trial) ;
    choiceunCh = 3-choice;

    idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ; % 1-3
    idx_color(2)    = colorMap(inputTarget(2, cnt_trial))+3 ; % 4-6
    idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial))+6 ; % 7-9
    idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
    idx_color(1)    = colorMap(inputTarget(1, cnt_trial))+3 ;
    idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial))+6 ;
    idx_patternshape(1) = (idx_pattern(1)-7)*3 + idx_shape(1) ; % 1-9
    idx_patternshape(2) = (idx_pattern(2)-7)*3 + idx_shape(2) ; 
    idx_patterncolor(1) = (idx_pattern(1)-7)*3 + idx_color(1)-3+9 ; % 10-18
    idx_patterncolor(2) = (idx_pattern(2)-7)*3 + idx_color(2)-3+9 ;
    idx_shapecolor(1) = (idx_shape(1)-1)*3 + idx_color(1)+18-3 ;
    idx_shapecolor(2) = (idx_shape(2)-1)*3 + idx_color(2)+18-3 ; % 19-27

    attn_w_feat_choice = attention_weights(vf, ...
        [idx_shape(1), idx_color(1), idx_pattern(1)], ...
        [idx_shape(2), idx_color(2), idx_pattern(2)], ...
        sesdata.attn_mode_choice, beta_attn_choice_feat);

    attn_w_conj_choice = attention_weights(vc, ...
        [idx_patternshape(1), idx_patterncolor(1), idx_shapecolor(1)], ...
        [idx_patternshape(2), idx_patterncolor(2), idx_shapecolor(2)], ...
        sesdata.attn_mode_choice, beta_attn_choice_conj);

    pChoiceR = 1./(1+exp(-magF*(attn_w_feat_choice(1)*(vf(idx_shape(2))-vf(idx_shape(1))) ...
                               +attn_w_feat_choice(2)*(vf(idx_color(2))-vf(idx_color(1))) ...
                               +attn_w_feat_choice(3)*(vf(idx_pattern(2))-vf(idx_pattern(1))))...
                         -magC*(attn_w_conj_choice(1)*(vc(idx_patternshape(2))-vc(idx_patternshape(1))) ...
                               +attn_w_conj_choice(2)*(vc(idx_patterncolor(2))-vc(idx_patterncolor(1))) ...
                               +attn_w_conj_choice(3)*(vc(idx_shapecolor(2))-vc(idx_shapecolor(1)))) ...
                         +BiasL)) ;

    pChoiceL = 1-pChoiceR ;
    if cnt_trial >= 1
        if choice == 2
            loglikehood(cnt_trial) = - log(pChoiceR) ;
        else
            loglikehood(cnt_trial) = - log(pChoiceL) ;
        end
    end

    attn_w_feat_learn = attention_weights(vf, ...
        [idx_shape(1), idx_color(1), idx_pattern(1)], ...
        [idx_shape(2), idx_color(2), idx_pattern(2)], ...
        sesdata.attn_mode_learn, beta_attn_learn_feat);

    attn_w_conj_learn = attention_weights(vc, ...
        [idx_patternshape(1), idx_patterncolor(1), idx_shapecolor(1)], ...
        [idx_patternshape(2), idx_patterncolor(2), idx_shapecolor(2)], ...
        sesdata.attn_mode_learn, beta_attn_learn_conj);

    % conjunction
    if correct
        idxC = idx_patternshape(choice) ;
        idxW = idx_patternshape(3-choice) ;
        vc = decayV(vc, find([1:9]~=idx_patternshape(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(1)) ;
        
        idxC = idx_patterncolor(choice) ;
        idxW = idx_patterncolor(3-choice) ;
        vc = decayV(vc, 9+find(9+[1:9]~=idx_patterncolor(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(2)) ;
        
        idxC = idx_shapecolor(choice) ;
        idxW = idx_shapecolor(3-choice) ;
        vc = decayV(vc, 18+find(18+[1:9]~=idx_shapecolor(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(3)) ;
    else
        idxC = idx_patternshape(3-choice) ;
        idxW = idx_patternshape(choice) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = decayV(vc, find([1:9]~=idx_patternshape(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(1)) ;
        
        idxC = idx_patterncolor(3-choice) ;
        idxW = idx_patterncolor(choice) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = decayV(vc, 9+find(9+[1:9]~=idx_patterncolor(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(2)) ;
        
        idxC = idx_shapecolor(3-choice) ;
        idxW = idx_shapecolor(choice) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = decayV(vc, 18+find(18+[1:9]~=idx_shapecolor(choice)), decay) ;
        [idxW, idxC] = idxcouple(idxW, idxC, correct, 0) ;
        vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(3)) ;
    end
    if flag_couple
        if correctunCh
            idxC = idx_patternshape(choiceunCh, cnt_trial) ;
            idxW = idx_patternshape(3-choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(1)) ;
            idxC = idx_patterncolor(choiceunCh, cnt_trial) ;
            idxW = idx_patterncolor(3-choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(2)) ;
            idxC = idx_shapecolor(choiceunCh, cnt_trial) ;
            idxW = idx_shapecolor(3-choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_rew*attn_w_conj_learn(3)) ;
        else
            idxC = idx_patternshape(3-choiceunCh, cnt_trial) ;
            idxW = idx_patternshape(choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(1)) ;
            idxC = idx_patterncolor(3-choiceunCh, cnt_trial) ;
            idxW = idx_patterncolor(choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(2)) ;
            idxC = idx_shapecolor(3-choiceunCh, cnt_trial) ;
            idxW = idx_shapecolor(choiceunCh, cnt_trial) ;
            [idxW, idxC] = idxcouple(idxW, idxC, correctunCh, 0) ;
            vc = update(vc, idxC, idxW, alpha_unr*attn_w_conj_learn(3)) ;
        end
    end

    % feature
    if correct
        idxC = idx_color(choice) ;
        idxW = idx_color(3-choice) ;
        vf   = decayV(vf, 3+find(3+[1:3]~=idx_color(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_rewColor*attn_w_feat_learn(1)) ;
        idxC = idx_shape(choice) ;
        idxW = idx_shape(3-choice) ;
        vf   = decayV(vf, find([1:3]~=idx_shape(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_rewShape*attn_w_feat_learn(2)) ;
        idxC = idx_pattern(choice) ;
        idxW = idx_pattern(3-choice) ;
        vf   = decayV(vf, 6+find(6+[1:3]~=idx_pattern(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_rewPattern*attn_w_feat_learn(3)) ;
    else
        idxW = idx_color(choice) ;
        idxC = idx_color(3-choice) ;
        vf   = decayV(vf, 3+find(3+[1:3]~=idx_color(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_unrColor*attn_w_feat_learn(1)) ;
        idxW = idx_shape(choice) ;
        idxC = idx_shape(3-choice) ;
        vf   = decayV(vf, find([1:3]~=idx_shape(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_unrShape*attn_w_feat_learn(2)) ;
        idxW = idx_pattern(choice) ;
        idxC = idx_pattern(3-choice) ;
        vf   = decayV(vf, 6+find(6+[1:3]~=idx_pattern(choice)), decay) ;
        [idxW, idxC] = idxcoupleF(idxW, idxC, correct, 0, flag_updatesim) ;
        vf           = update(vf, idxC, idxW, alpha_unrPattern*attn_w_feat_learn(3)) ;
    end
    if flag_couple
        if correctunCh
            idxC = idx_color(choiceunCh) ;
            idxW = idx_color(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_rewColor*attn_w_feat_learn(1)) ;
            idxC = idx_shape(choiceunCh) ;
            idxW = idx_shape(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_rewShape*attn_w_feat_learn(2)) ;
            idxC = idx_pattern(choiceunCh) ;
            idxW = idx_pattern(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_rewPattern*attn_w_feat_learn(3)) ;
        else
            idxW = idx_color(choiceunCh) ;
            idxC = idx_color(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_unrColor*attn_w_feat_learn(1)) ;
            idxW = idx_shape(choiceunCh) ;
            idxC = idx_shape(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_unrShape*attn_w_feat_learn(2)) ;
            idxW = idx_pattern(choiceunCh) ;
            idxC = idx_pattern(3-choiceunCh) ;
            [idxW, idxC] = idxcoupleF(idxW, idxC, correctunCh, 0, flag_updatesim) ;
            vf = update(vf, idxC, idxW, alpha_unrPattern*attn_w_feat_learn(3)) ;
        end
    end
    V(1:9,cnt_trial) = vf ;
    V(10:36,cnt_trial) = vc ;
    A(1,1:3,cnt_trial) = attn_w_feat_choice;
    A(2,1:3,cnt_trial) = attn_w_feat_learn;
    A(1,4:6,cnt_trial) = attn_w_conj_choice;
    A(2,4:6,cnt_trial) = attn_w_conj_learn;
end
end

% function v = decayV(v, unCh, decay)
%     v(unCh) = v(unCh)*(1-decay) ;
% end
function v = decayV(v, unCh, decay)
v(unCh) = v(unCh) - (v(unCh)-0.5)*(decay) ;
end

function v = update(v, idxC, idxW, Q)
if isempty(idxW)
    v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
elseif isempty(idxC)
    v(idxW) = v(idxW) - (v(idxW).*Q) ;
elseif ~isempty(idxW) && ~isempty(idxC)
    v(idxC) = v(idxC) + (1-v(idxC)).*Q ;
    v(idxW) = v(idxW) - (v(idxW).*Q) ;
end
end

function [idxW, idxC] = idxcouple(idxW, idxC, rl2_correct, flag_couple)
if rl2_correct
    if flag_couple==0
        idxW = [] ;
    end
else
    if flag_couple==0
        idxC = [] ;
    end
end
end

function [idxW, idxC] = idxcoupleF(idxW, idxC, rl2_correct, flag_couple, flag_updatesim)
if rl2_correct
    if flag_couple==0
        idxW = [] ;
    elseif flag_couple==1
        if idxW==idxC                                                  % to avoid potentiating and depressing similar V in coupled cases
            idxW= [] ;
            if ~flag_updatesim
                idxC = [] ;
            end
        end
    end
else
    if flag_couple==0
        idxC = [] ;
    elseif flag_couple==1
        if idxW==idxC                                                  % to avoid potentiating and depressing similar V in coupled cases
            idxC= [] ;
            if ~flag_updatesim
                idxW = [] ;
            end
        end
    end
end
end

function [attn] = attention_weights(v, idxes1, idxes2, mode, beta)
if strcmp(mode, 'diff')
    attn = softmax(beta*abs(v(idxes1)-v(idxes2)));
elseif strcmp(mode, 'sum')
    attn = softmax(beta*(v(idxes1)+v(idxes2)));
elseif strcmp(mode, 'max')
    attn = softmax(beta*max(v(idxes1), v(idxes2)));
else
    attn = ones(size(idxes1))/ones(size(idxes1));
end
end