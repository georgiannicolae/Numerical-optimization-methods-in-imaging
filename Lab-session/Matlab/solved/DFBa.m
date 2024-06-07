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

NL = 1;
gamma = 1/NL;
zeta = 2.1;
Yold = Y;
costold = -1;

for nit = 1:nitm
    zeta = (nit-1)/(nit+zeta);
    Z = Y+zeta*(Y-Yold);
    X = Xtilde-Z;
    X(X<0) = 0; % nonnegativity constraint

    Yt = Z+gamma*X;
    [U,D,V] = svd(Yt);
    dthresh = max((diag(D)-lambda)/gamma,0);
    PYt = U*diag(dthresh)*V'; % nuclear norm penalization
    Yold = Y;
    Y =  Yt-gamma*PYt;

    [U,D,V] = svd(X);
    cost(nit) = norm(X-Xtilde,'fro')^2/2+lambda*sum(D(:));
    fprintf(1,'%d: cost= %g\n',nit,cost(nit));

    if nit > 1 && abs(cost(nit)-costold) < prec*costold
        break
    end
    costold = cost(nit);
end