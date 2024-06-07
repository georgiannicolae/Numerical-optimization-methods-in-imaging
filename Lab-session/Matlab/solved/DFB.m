function [X,Y,cost, nit] = DFB(Xtilde,lambda,nitm,prec)
%[X,Y] = DFB(Xtilde,lambda,nitm): Dual Forward-Backward algorithm
% Xtilde : reference matrix
% lambda : nuclear norm regularization parameter
% nitm : maximum number of iterations
% prec : precision for stopping test
% X : output primal variable
% Y : output dual variable
% cost : cost values along iteration

Y = zeros(size(Xtilde));

NL = 1;
gamma = 1.9/NL;

costold = -1;
for nit = 1:nitm
    X = Xtilde-Y;
    X(X<0) = 0; % nonnegativity constraint

    Z = Y+gamma*X;
    [U,D,V] = svd(Z);
    dthresh = max((diag(D)-lambda)/gamma,0);
    PZ = U*diag(dthresh)*V'; % nuclear norm penalization
    Y =  Z-gamma*PZ;

    [U,D,V] = svd(X);
    cost(nit) = norm(X-Xtilde,'fro')^2/2+lambda*sum(D(:));
    fprintf(1,'%d: cost= %g\n',nit,cost(nit));

    if nit > 1 && abs(cost(nit)-costold) < prec*costold
        break
    end
    costold = cost(nit);
end

