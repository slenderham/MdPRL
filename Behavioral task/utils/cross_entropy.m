function ce = cross_entropy(ps, ts)
ce = -sum(ts.*log(ps+1E-5),2);
end