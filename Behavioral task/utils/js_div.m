function dist=js_div(P,Q,dim)
%  dist = KLDiv(P,Q) Jensen-Shannon divergence of two discrete probability
%  distributions
%  P and Q  should have the sum of one on last dimension
%  P =  (... x nbins)
%  Q =  (... x nbins)
%  dist = (...)

% normalizing the P and Q

M = (P+Q)./2;
dist = 0.5*P.*(log(P+1e-10)-log(M)) + 0.5*Q.*(log(Q+1e-10)-log(M));
dist(isnan(dist))=0; % resolving the case when P(i)==0
dist = sum(dist, dim);
end
