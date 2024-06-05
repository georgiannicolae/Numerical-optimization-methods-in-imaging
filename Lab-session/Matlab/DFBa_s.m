function [X,Y,cost, nit] = DFBa(Xtilde,lambda,nitm,prec)
%[X,Y] = DFBa(Xtilde,lambda,nitm): accelerated Dual Forward-Backward algorithm
% Xtilde : reference matrix
% lambda : nuclear norm regularization parameter
% nitm : maximum number of iterations
% prec : precision for stopping test
% X : output primal variable
% Y : output dual variable
% cost : cost values along iteration

Y = zeros(size(Xtilde));
Z = Y;
gamma = 1.99;
zeta = 2.1;
Yold = Y;
costold = -1;

for nit = 1:nitm
% start your code 


% end your code
    if nit > 1 && abs(cost(nit)-costold) < prec*costold
        break
    end
    costold = cost(nit);
end