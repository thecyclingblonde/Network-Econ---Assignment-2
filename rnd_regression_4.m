% Logistic regression to predict links in the following SAR models:
% model 1: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm fixed effects)
% model 2: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with time fixed effects)
% model 3: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm and time fixed effects)

clear

yr0 = 1962;
yr1 = 1966;
yr2 = 2006;
lnkyear = 5;

%% Load data.
load('./Data/ID.mat') % firm ids
load('./Data/dm.mat') % missing data indicator
load('./Data/X1.mat') % firm covariates
load('./Data/X2.mat') % aggr
load('./Data/X3.mat') % A*firm
load('./Data/NAT.mat') % geographic locations

n = length(ID(:,1));

% drop the first t0 year(s)
t0 = 1;
T = yr2-yr1+1;
dt = cat(1,dm{:});
dt = dt(n*t0+1:end);
T1 = T-t0;

%% Drop firms that appear less than twice.
di = (1:n*T1)';
dj = kron(ones(T1,1),(1:n)');
Dn = sparse(di,dj,dt);
dd = sum(Dn,1);
Dn = Dn(:,dd>1); % firms appearing more than once in the panel
np = size(Dn,2);

E  = speye(n);
E  = E(dd>1,:); % firms appearing more than once in the panel
ID = E*ID;
for s = 1:T
    dm{s} = E*dm{s};
    X1{s} = E*X1{s};
    X2{s} = E*X2{s};
    X3{s} = E*X3{s};
end

%% Construct data matrices
di  = cell(T1,1);
dj1 = cell(T1,1);
dj2 = cell(T1,1);
dj3 = cell(T1,1);
n1 = 0;
n2 = 0;
for s = 1:T1
    D = Dn(n*(s-1)+1:n*s,:);
    D = D(sum(D,2)==1,:); % firms without missing observations
    [ni,nj] = size(D); % note: ni is the number of nonzero entries of D
    [ii,jj,~] = find(D);
    di{s}  = n1+ii;
    dj1{s} = jj;
    dj2{s} = s*ones(ni,1);
    dj3{s} = n2+jj;
    
    n1 = n1+ni;
    n2 = n2+nj;
end
di  = cat(1,di{:});
dj1 = cat(1,dj1{:});
dj2 = cat(1,dj2{:});
dj3 = cat(1,dj3{:});

D1 = sparse(di,dj1,ones(n1,1),n1,np);
D2 = sparse(di,dj2,ones(n1,1),n1,T1); % block diagonal matrix of D*1
D3 = sparse(di,dj3,ones(n1,1),n1,n2); % block diagonal matrix of D

L1 = D2(:,2:end); % time dummies

FID = D3*kron(ones(T1,1),ID(:,1));
UID = unique(FID);
L2 = bsxfun(@eq,FID,UID(2:end)'); % firm dummies

DD1 = (D1'*D1)\D1';
DD1 = speye(n1)-D1*DD1;
PD1 = DD1*D3;

DD2 = (D2'*D2)\D2';
DD2 = speye(n1)-D2*DD2;
PD2 = DD2*D3;

D4  = DD1*D2;
DD4 = (D4'*D4)\D4';
DD4 = D4*DD4;
PD3 = (DD1-DD4)*D3;

%% Load adjacency matrices and patent proximity matrices.
ID3 = floor(ID(:,2)/100);
grp = bsxfun(@eq,ID3,ID3');
grp(triu(true(size(grp))))=0;
[gi,gj,~] = find(grp);
nl = length(gi);
T2 = yr2-yr0+1;
Wy = zeros(nl,T2);
Wx = zeros(nl,T2);
Wa = zeros(nl,T2);
Wb = zeros(nl,T2);
W1 = zeros(nl,T2);
W2 = zeros(nl,T2);
for s = 1:T2
    yr = yr0+s-1;
    
    load(['./Data/A_' int2str(yr) '.mat']) % Load adjacency matrix.
    B = double(A^2>0)-double(A>0); % Second order neighbors
    load(['./Data/P_' int2str(yr) '.mat']) % Load technology proximity matrix.

    Wy(:,s) = A(grp==1);
    Wx(:,s) = B(grp==1);
    Wa(:,s) = sum(Wy(:,1:s),2);
    Wb(:,s) = sum(Wx(:,1:s),2);
    W1(:,s) = P(grp==1);
end

%% Adjust location data.
tmp1 = NAT(:,1);
tmp2 = NAT(:,2);
tmp3 = NAT(:,3);
loc = zeros(np,2);
for i = 1:np
    if sum(tmp1==ID(i,1)) == 1
        loc(i,1) = tmp2(tmp1==ID(i,1));
        loc(i,2) = tmp3(tmp1==ID(i,1));
    end
end

%% Compute geographic distances.
dis = pdist2(loc,loc); % Pairwise distance between two sets of observations
prx = zeros(np);
prx(dis==0) = 1;
dc = prx(grp==1);
dc = kron(ones(T1,1),dc);

%% Construct data matrices for logistic regression.
n0 = nl*T1;
dy = reshape(Wy(:,T2-T1+1:T2),n0,1);
da = reshape(Wa(:,T2-T1-lnkyear+1:T2-lnkyear),n0,1);
db = reshape(Wb(:,T2-T1-lnkyear+1:T2-lnkyear),n0,1);

dp = reshape(W1(:,T2-T1-lnkyear+1:T2-lnkyear),n0,1);

dx = [da,db,dp,dp.^2,dc,ones(n0,1)];
kx = size(dx,2);

%% Estimate logistic regression parameters.
xx = dx'*dx;
b0 = xx\(dx'*dy);
option = optimoptions(@fminunc,'Algorithm','trust-region','GradObj','on','Hessian','on','Display','notify','DerivativeCheck','off');
[b0,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc('logit_obj',b0,option,dy,dx);
s0 = sqrt(diag(HESSIAN\speye(kx)))';

p0 = exp(dx*b0)./(1+exp(dx*b0));
d0 = 0;
for i = 1:n0
    d0 = d0+dy(i)*log(p0(i))+(1-dy(i))*log(1-p0(i));
end
y0 = mean(dy);
d1 = n0*(y0*log(y0)+(1-y0)*log(1-y0));
R2 = 1-d0/d1; % McFadden's R-squared (Cameron and Trivedi, p.474)
%http://www.ats.ucla.edu/stat/mult_pkg/faq/general/Psuedo_RSquareds.htm

%% Construct prediceted adjacency matrix.
for s = (t0+1):T
    Ap = sparse(gi,gj,p0(nl*(s-t0-1)+1:nl*(s-t0)),np,np);
    Ap = Ap+Ap';
    X3{s}(:,2) = Ap*X1{s}(:,2);
end

%% Construct data matrices for SAR models.
Yn = cell(T1,1);
Zn = cell(T1,1);
Qn = cell(T1,1);
for s = (t0+1):T
    Yn{s-t0} = X1{s}(:,1);
    Zn{s-t0} = [X3{s}(:,1),X2{s}(:,1),X1{s}(:,2)];
    Qn{s-t0} = [X3{s}(:,2),X2{s}(:,2),X1{s}(:,2)];
end
Yn = cat(1,Yn{:});
Zn = cat(1,Zn{:});
Qn = cat(1,Qn{:});

nk = size(Zn,2);

%% 2SLS estimation of model 1: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm fixed effects)
Y1 = PD1*Yn;
Z1 = PD1*Zn;
Q1 = PD1*Qn;
QQ = Q1'*Q1;
QZ = Q1'*Z1;
QY = Q1'*Y1;
PI = QQ\QZ;
ZZ = (QZ'*PI)\PI';
b1 = ZZ*QY;
% robust s.e.
u1 = Y1-Z1*b1;
V1 = Q1'*sparse(1:n1,1:n1,u1.^2)*Q1;
s1 = sqrt(spdiags(ZZ*V1*ZZ',0));

%% 2SLS estimation of model 2: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with time fixed effects)
Y2 = PD2*Yn;
Z2 = PD2*Zn;
Q2 = PD2*Qn;
QQ = Q2'*Q2;
QZ = Q2'*Z2;
QY = Q2'*Y2;
PI = QQ\QZ;
ZZ = (QZ'*PI)\PI';
b2 = ZZ*QY;
% robust s.e.
u2 = Y2-Z2*b2;
V2 = Q2'*sparse(1:n1,1:n1,u2.^2)*Q2;
s2 = sqrt(spdiags(ZZ*V2*ZZ',0));

%% 2SLS estimation of model 3: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm and time fixed effects)
Y3 = PD3*Yn;
Z3 = PD3*Zn;
Q3 = PD3*Qn;
QQ = Q3'*Q3;
QZ = Q3'*Z3;
QY = Q3'*Y3;
PI = QQ\QZ;
ZZ = (QZ'*PI)\PI';
b3 = ZZ*QY;
% robust s.e.
u3 = Y3-Z3*b3;
V3 = Q3'*sparse(1:n1,1:n1,u3.^2)*Q3;
s3 = sqrt(spdiags(ZZ*V3*ZZ',0));

%% Print results.
fid = fopen(['./output4.txt'],'w');
fprintf(fid,'Logistic regression \n');
coeff = {'da' 'db' 'pat' 'pat.^2' 'dc' 'const' };
for ip = 1:kx
    tstat = b0(ip)/s0(ip);
    tstat = abs(tstat);
    if tstat >= 2.326
        fprintf(fid,'%s: %7.4f*** (%6.4f)\n',coeff{ip},b0(ip),s0(ip));
    elseif tstat >= 1.96
        fprintf(fid,'%s: %7.4f** (%6.4f)\n',coeff{ip},b0(ip),s0(ip));
    elseif tstat >= 1.645
        fprintf(fid,'%s: %7.4f* (%6.4f)\n',coeff{ip},b0(ip),s0(ip));
    else
        fprintf(fid,'%s: %7.4f (%6.4f)\n',coeff{ip},b0(ip),s0(ip));
    end
end
fprintf(fid,'Link prediction R^2 is %7.4f \n',R2);
fprintf(fid,'\n');
fprintf(fid,'SAR with firm fixed effects \n');
coeff = {'lambda' 'rho   ' 'beta  ' };
for ip = 1:nk
    tstat = b1(ip)/s1(ip);
    tstat = abs(tstat);
    if tstat >= 2.326
        fprintf(fid,'%s: %7.4f*** (%6.4f)\n',coeff{ip},b1(ip),s1(ip));
    elseif tstat >= 1.96
        fprintf(fid,'%s: %7.4f** (%6.4f)\n',coeff{ip},b1(ip),s1(ip));
    elseif tstat >= 1.645
        fprintf(fid,'%s: %7.4f* (%6.4f)\n',coeff{ip},b1(ip),s1(ip));
    else
        fprintf(fid,'%s: %7.4f (%6.4f)\n',coeff{ip},b1(ip),s1(ip));
    end
end
fprintf(fid,'\n');
fprintf(fid,'SAR with time fixed effects \n');
for ip = 1:nk
    tstat = b2(ip)/s2(ip);
    tstat = abs(tstat);
    if tstat >= 2.326
        fprintf(fid,'%s:  %7.4f*** (%6.4f)\n',coeff{ip},b2(ip),s2(ip));
    elseif tstat >= 1.96
        fprintf(fid,'%s:  %7.4f** (%6.4f)\n',coeff{ip},b2(ip),s2(ip));
    elseif tstat >= 1.645
        fprintf(fid,'%s:  %7.4f* (%6.4f)\n',coeff{ip},b2(ip),s2(ip));
    else
        fprintf(fid,'%s:  %7.4f (%6.4f)\n',coeff{ip},b2(ip),s2(ip));
    end
end
fprintf(fid,'\n');
fprintf(fid,'SAR with firm and time fixed effects \n');
for ip = 1:nk
    tstat = b3(ip)/s3(ip);
    tstat = abs(tstat);
    if tstat >= 2.326
        fprintf(fid,'%s:  %7.4f*** (%6.4f)\n',coeff{ip},b3(ip),s3(ip));
    elseif tstat >= 1.96
        fprintf(fid,'%s:  %7.4f** (%6.4f)\n',coeff{ip},b3(ip),s3(ip));
    elseif tstat >= 1.645
        fprintf(fid,'%s:  %7.4f* (%6.4f)\n',coeff{ip},b3(ip),s3(ip));
    else
        fprintf(fid,'%s:  %7.4f (%6.4f)\n',coeff{ip},b3(ip),s3(ip));
    end
end
fclose(fid);