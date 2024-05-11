function y = pseudo_log(x, base)

if nargin==1
    base = exp(1);
end

y = asinh(x/2)/log(base);
end