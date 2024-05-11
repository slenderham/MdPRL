function y = logsigmoid(x)
    y(x>0) = -log(1+exp(-x(x>0)));
    y(x<=0) = x(x<=0)-log(1+exp(x(x<=0)));
end