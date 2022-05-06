function lme = laplace_approximation(lls, priors, hessians, K)

    symm_hessian = (hessians+hessians')/2;
    h = log(det(symm_hessian)); 
    lme = lls + priors + 0.5*(K*log(2*pi) - h);

end
