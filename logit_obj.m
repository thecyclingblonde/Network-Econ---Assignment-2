function [f,g,H] = logit_obj(b,Y,X)
% f: function value
% g: gradient
% H: hessian

n = length(Y);

u = exp(X*b);
P = u./(1+u);

f = Y.*log(P)+(1-Y).*log(1-P);
f = -sum(f);

g = X'*(Y-P);
g = -g;

H = -X'*sparse(1:n,1:n,P.*(1-P))*X;
H = -H;