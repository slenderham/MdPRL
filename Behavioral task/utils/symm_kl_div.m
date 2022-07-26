function dist=symm_kl_div(P,Q,dim)
%  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
%  distributions
%  P and Q  should have the sum of one on last dimension
%  P =  (... x nbins)
%  Q =  (... x nbins)
%  dist = (...)

% normalizing the P and Q

dist = (P.*(log(P+1e-10)-log(Q+1e-10)) + Q.*(log(Q+1e-10)-log(P+1e-10)));
dist(isnan(dist))=0; % resolving the case when P(i)==0
dist = sum(dist, dim);
end
