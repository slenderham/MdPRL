function stars = sig2ast(p)

stars = string(p);
stars(p<1E-3)='***';
stars((p>=1E-3)&(p<1E-2))='**';
stars((p>=1E-2)&(p<0.05))='*';
stars((p>=0.05)&(p<0.1))='+';
stars(p>=0.10)='';