function [loglikehood, latents] = fMLchoiceLL_RW_conjdecayattn_constrained(xpar, sesdata)
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

BiasL = xpar(1) ;
mag  = xpar(2) ;
omega  = xpar(3) ;
decay = xpar(4) ;

magF = mag*omega;
magC = mag*(1-omega);

alpha_rew           = xpar(NparamBasic+1) ;
NparamWithLR = NparamBasic+1;

if sesdata.flagUnr==1
    alpha_unr           = xpar(NparamBasic+2) ;
    NparamWithLR = NparamBasic+2;
else
    alpha_unrColor      = alpha_rewColor ;
    alpha_unrShape      = alpha_rewShape ;
    alpha_unrPattern    = alpha_rewPattern ;
    alpha_unr           = alpha_rew ;
end

if strcmp(sesdata.attn_time, "none")
    beta_attn_feat = 0;
    beta_attn_conj = 0;
else
    beta_attn_feat = xpar(NparamWithLR+1)*omega;
    beta_attn_conj = xpar(NparamWithLR+1)*(1-omega);
end

% input to feature maps
shapeMap        = sesdata.expr.shapeMap ;
colorMap        = sesdata.expr.colorMap ;
patternMap      = sesdata.expr.patternMap ;

% S(tates)
inputTarget     = sesdata.input.inputTarget ;
% R(ewards)
correcttrials   = sesdata.results.reward ;
% A(ctions)
choicetrials    = sesdata.results.choice ;

% model specs
flag_couple     = sesdata.flag_couple ;
flag_updatesim  = sesdata.flag_updatesim ;
ntrials         = length(choicetrials) ;
inputRewards    = sesdata.input.inputReward ;

vf              = (0.5*ones(9,1)) ; % 1-3 shape, 4-6 color, 7-9 pattern
vc              = (0.5*ones(27,1)) ; % 1-9 pXs, 10-18 pXc, 19-27 sXc

for cnt_trial=1:ntrials
    latents.V(1:9,cnt_trial) = vf ;
    latents.V(10:36,cnt_trial) = vc ;
    % current trial C and R
    correct = correcttrials(cnt_trial) ;
    choice = choicetrials(cnt_trial) ;
    if ~isnan(choice) && ~isnan(correct)
        correctunCh = inputRewards(3-choice, cnt_trial) ;
        choiceunCh = 3-choice ;
    end

    % current trial S
    idx_shape(2)    = shapeMap(inputTarget(2, cnt_trial)) ; % 1-3
    idx_color(2)    = colorMap(inputTarget(2, cnt_trial))+3 ; % 4-6
    idx_pattern(2)  = patternMap(inputTarget(2, cnt_trial))+6 ; % 7-9
    idx_shape(1)    = shapeMap(inputTarget(1, cnt_trial)) ;
    idx_color(1)    = colorMap(inputTarget(1, cnt_trial))+3 ;
    idx_pattern(1)  = patternMap(inputTarget(1, cnt_trial))+6 ;
    idx_patternshape(1) = (idx_pattern(1)-7)*3 + idx_shape(1) ; % 1-9
    idx_patternshape(2) = (idx_pattern(2)-7)*3 + idx_shape(2) ; 
    assert(1<=idx_patternshape(1) & idx_patternshape(1)<=9 & 1<=idx_patternshape(2) & idx_patternshape(2)<=9);
    idx_patterncolor(1) = (idx_pattern(1)-7)*3 + (idx_color(1)-4)+10 ; % 10-18
    idx_patterncolor(2) = (idx_pattern(2)-7)*3 + (idx_color(2)-4)+10 ;
    assert(10<=idx_patterncolor(1) & idx_patterncolor(1)<=18 & 10<=idx_patterncolor(2) & idx_patterncolor(2)<=18);
    idx_shapecolor(1) = (idx_shape(1)-1)*3 + (idx_color(1)-4)+19 ;
    idx_shapecolor(2) = (idx_shape(2)-1)*3 + (idx_color(2)-4)+19 ; % 19-27
    assert(19<=idx_shapecolor(1) & idx_shapecolor(1)<=27 & 19<=idx_shapecolor(2) & idx_shapecolor(2)<=27);

    attn_w = attention_weights( ...
                    beta_attn_feat*vf([idx_shape(1), idx_color(1), idx_pattern(1)])...
                   +beta_attn_conj*vc([idx_patterncolor(1), idx_patternshape(1), idx_shapecolor(1)]), ...
                    beta_attn_feat*vf([idx_shape(2), idx_color(2), idx_pattern(2)])...
                   +beta_attn_conj*vc([idx_patterncolor(2), idx_patternshape(2), idx_shapecolor(2)]), ...
                    sesdata.attn_op, 1);

    if strcmp(sesdata.attn_time, "C")
        attn_w_choice = attn_w;
        attn_w_learn = ones(1, 3)/3;
    elseif strcmp(sesdata.attn_time, "L")
        attn_w_choice = ones(1, 3)/3;
        attn_w_learn = attn_w;
    elseif strcmp(sesdata.attn_time, "CL")
        attn_w_choice = attn_w;
        attn_w_learn = attn_w;
    elseif strcmp(sesdata.attn_time, "none")
        attn_w_choice = ones(1, 3)/3;
        attn_w_learn = ones(1, 3)/3;
    end

    vo(1) = attn_w_choice(1)*(omega*vf(idx_shape(1))+(1-omega)*vc(idx_patterncolor(1)))+...
            attn_w_choice(2)*(omega*vf(idx_color(1))+(1-omega)*vc(idx_patternshape(1)))+...
            attn_w_choice(3)*(omega*vf(idx_pattern(1))+(1-omega)*vc(idx_shapecolor(1)));

    vo(2) = attn_w_choice(1)*(omega*vf(idx_shape(2))+(1-omega)*vc(idx_patterncolor(2)))+...
            attn_w_choice(2)*(omega*vf(idx_color(2))+(1-omega)*vc(idx_patternshape(2)))+...
            attn_w_choice(3)*(omega*vf(idx_pattern(2))+(1-omega)*vc(idx_shapecolor(2)));

    logit = mag*(vo(2)-vo(1))-BiasL;
 
    if cnt_trial >= 1
        % if choice is nan then sample the choice
        if isnan(choice) || isnan(correct)
            pChoiceR = 1./(1+exp(-logit));
        
            choice = binornd(1, pChoiceR)+1;
            choiceunCh = 3-choice;
            
            correct = inputRewards(choice, cnt_trial) ;
            correctunCh = inputRewards(3-choice, cnt_trial) ;
        end
        latents.R(cnt_trial) = correct;
        rpe = correct-vo(choice);
        if rpe>0
            alpha = alpha_rew;
        else
            alpha = alpha_unr;
        end
        latents.C(cnt_trial) = choice;
        latents.logits(cnt_trial) = logit;
        if choice == 2
            loglikehood(cnt_trial) =  - logsigmoid(logit) ;
        else
            loglikehood(cnt_trial) =  - logsigmoid(-logit) ;
        end
    end

    % conjunction
    idxC = idx_patternshape(choice) ;
    vc = decayV(vc, find([1:9]~=idx_patternshape(choice)), decay) ;
    vc = update(vc, idxC, rpe, alpha*attn_w_learn(2)) ;
    
    idxC = idx_patterncolor(choice) ;
    vc = decayV(vc, 9+find(9+[1:9]~=idx_patterncolor(choice)), decay) ;
    vc = update(vc, idxC, rpe, alpha*attn_w_learn(1)) ;
    
    idxC = idx_shapecolor(choice) ;
    vc = decayV(vc, 18+find(18+[1:9]~=idx_shapecolor(choice)), decay) ;
    vc = update(vc, idxC, rpe, alpha*attn_w_learn(3)) ;

    % feature
    idxC = idx_color(choice) ;
    vf = decayV(vf, 3+find(3+[1:3]~=idx_color(choice)), decay) ;
    vf = update(vf, idxC, rpe, alpha*attn_w_learn(2)) ;
    
    idxC = idx_shape(choice) ;
    vf = decayV(vf, find([1:3]~=idx_shape(choice)), decay) ;
    vf = update(vf, idxC, rpe, alpha*attn_w_learn(1)) ;
    
    idxC = idx_pattern(choice) ;
    vf  = decayV(vf, 6+find(6+[1:3]~=idx_pattern(choice)), decay) ;
    vf  = update(vf, idxC, rpe, alpha*attn_w_learn(3)) ;
    
    latents.A(1,1:3,cnt_trial) = attn_w_choice;
    latents.A(2,1:3,cnt_trial) = attn_w_learn;
end
end

% function v = decayV(v, unCh, decay)
%     v(unCh) = v(unCh)*(1-decay) ;
% end
function v = decayV(v, unCh, decay)
    v(unCh) = v(unCh) - (v(unCh)-0.5)*decay;
end

function v = update(v, idxC, rpe, Q)
    v(idxC) = v(idxC) + rpe.*Q ;
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
        if idxW==idxC               % to avoid potentiating and depressing similar V in coupled cases
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
        if idxW==idxC               % to avoid potentiating and depressing similar V in coupled cases
            idxC= [] ;
            if ~flag_updatesim
                idxW = [] ;
            end
        end
    end
end
end

function [attn] = attention_weights(v1, v2, mode, beta)
    if strcmp(mode, 'diff')
        attn = softmax(beta*abs(v1-v2));
    elseif strcmp(mode, 'sum')
        attn = softmax(beta*(v1+v2)/2);
    elseif strcmp(mode, 'max')
        attn = softmax(beta*max(v1, v2));
    elseif strcmp(mode, 'const')
        attn = ones(size(v1))./numel(v1);
    else
        error('attn mode not recognized');
    end
end