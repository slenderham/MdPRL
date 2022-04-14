function [expr] = fPRL2initExp(flaginf)
%%

% which feature is with information
if flaginf==1
    expr.targetPattern = repmat([1 2 3 ;
                    1 2 3 ;
                    1 2 3 ], 1,1,3) ;

    expr.targetShape = repmat([1 1 1 ;
                    2 2 2 ;
                    3 3 3], 1,1,3) ;
elseif flaginf==2
    expr.targetShape = repmat([1 2 3 ;
                    1 2 3 ;
                    1 2 3 ], 1,1,3) ;

    expr.targetPattern = repmat([1 1 1 ;
                    2 2 2 ;
                    3 3 3], 1,1,3) ;
end
                
expr.targetColor(:,:,1) = ones(3,3) ;
expr.targetColor(:,:,2) = 2*ones(3,3) ;
expr.targetColor(:,:,3) = 3*ones(3,3) ;

% target indices: 1:27
expr.playcombinations   = [1:27] ;
expr.choiceMap          = nchoosek(expr.playcombinations,2)' ;    
% we only show those that differ in all features
idx_targetColor         = find(diff(expr.targetColor(expr.choiceMap),1)) ;
idx_targetShape         = find(diff(expr.targetShape(expr.choiceMap),1)) ;
idx_targetPattern       = find(diff(expr.targetPattern(expr.choiceMap),1)) ;
idx_intersect           = intersect(idx_targetColor, idx_targetShape) ;
idx_intersect           = intersect(idx_targetPattern, idx_intersect) ;
expr.choiceMap          = expr.choiceMap(:,idx_intersect) ;
expr.choiceMap          = [expr.choiceMap expr.choiceMap([2 1],:)] ;

expr.Ncombinations      = length(expr.choiceMap) ;
expr.combinationMap     = [1:expr.Ncombinations] ;

% randomize shape/color/texture assigned to 27 targets
shapeperm               = randperm(3) ;
textureperm             = randperm(3) ;
colorperm               = randperm(3) ;
expr.shape              = shapeperm(1:3)-1 ;
expr.texture            = textureperm(1:3)-1 ;
expr.color              = colorperm(1:3) ;

expr.colorList          = colormap(parula(3)) ;
close 1

%%

expr.NtrialsShort   = expr.Ncombinations*2 ;
expr.Ntrials        = expr.NtrialsShort*1 ;
expr.Nblocks        = expr.Ntrials/expr.NtrialsShort ;
expr.NtrialsRewardAssign = expr.Ncombinations*2 ;                       % how often assign reward probabilities
expr.trialProbe     = [round(expr.Ncombinations*[2:2:10]*2/10)] ;

X   = 4 ; 
ll1 = [X      1  1/X;
       1/X^2  1  X^2;
       X      1  1/X]' ;
   
X   = 3 ; 
ll  = [X      1  1/X]' ;

LLSh            = nan*ones(3,3,3) ;
LLSh(1,:,:)     = ll1*ll(1) ;
LLSh(2,:,:)     = ones(size(ll1))*ll(2) ;
LLSh(3,:,:)     = ll1*ll(3) ;
expr.LL{1}      = LLSh ;

expr.prob{1}    = (expr.LL{1}./(1+expr.LL{1})) ;
for cnt_blocks  = 2:expr.Nblocks
    expr.prob{cnt_blocks}   = expr.prob{cnt_blocks-1} ;
    expr.LL{cnt_blocks}     = expr.LL{cnt_blocks-1} ;
end


end
