function ent = entropy(ps)
ent = -sum(ps.*log(ps+1E-8),2);
end