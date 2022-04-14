function MSE = fExpfit(xpar, data) 

y = data{1} ;
x = 1:length(y) ;

xpar = 1./(1+exp(-(xpar/0.2))) ;

cnst = xpar(1) ;
coeff = xpar(2) ; 
tau = xpar(3) ;

pchoice = cnst - coeff*exp(-x*tau) ;  
MSE = sum(abs(pchoice'-y)) ; 