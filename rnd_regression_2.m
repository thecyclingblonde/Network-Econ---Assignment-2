% model 2: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm fixed effects)

clear all
close all

%% Load data.
load('./Data/A.mat')  % adjacency matrix
load('./Data/ID.mat') % firm ids
load('./Data/dm.mat') % missing data indicator
load('./Data/X1.mat') % firm covariates
load('./Data/X2.mat') % aggr
load('./Data/X3.mat') % A*firm

n = length(A);

%% drop the first t0 years (i.e. the starting year is (yr1+t0))
yr1 = 1966;
yr2 = 2006;
t0 = 1;
T = yr2-yr1+1;
dt = cat(1,dm{:});
dt = dt(n*t0+1:end);
T1 = T-t0;

%% Drop firms that appear less than twice
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

di  = cell(T1,1);
dj1 = cell(T1,1);
n1 = 0;
n2 = 0;
for s = 1:T1
    D = Dn(n*(s-1)+1:n*s,:);
    D = D(sum(D,2)==1,:); % firms without missing observations
    [ni,nj] = size(D); % note: ni is the number of nonzero entries of D
    [ii,jj,~] = find(D);
    di{s}  = n1+ii;
    dj1{s} = jj;
    dj3{s} = n2+jj;
    
    n1 = n1+ni;
    n2 = n2+nj;
end
di  = cat(1,di{:});
dj1 = cat(1,dj1{:});
dj3 = cat(1,dj3{:});

D1 = sparse(di,dj1,ones(n1,1),n1,np);
D3 = sparse(di,dj3,ones(n1,1),n1,n2); % block diagonal matrix of D

%% Construct data matrices.
DD1 = (D1'*D1)\D1';
DD1 = speye(n1)-D1*DD1;
PD1 = DD1*D3;

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

%% 2SLS estimation of model 2: Y = lambda*A*Y + rho*Y_ttl + X*b + error (with firm fixed effects)
Y1 = PD1*Yn;
Z1 = PD1*Zn;
Q1 = PD1*Qn;
QQ = Q1'*Q1;
QZ = Q1'*Z1;
QY = Q1'*Y1;
PI = QQ\QZ;
ZZ = (QZ'*PI)\PI';
b = ZZ*QY;
% robust s.e.
u1 = Y1-Z1*b;
V1 = Q1'*sparse(1:n1,1:n1,u1.^2)*Q1;
se = sqrt(spdiags(ZZ*V1*ZZ',0));

%% Print results
fid = fopen(['./output2.txt'],'w');
%fprintf(fid,'with firm fixed effects \n');
coeff = {'lambda' 'rho   ' 'beta  ' };
for ip = 1:nk
    tstat = b(ip)/se(ip);
    tstat = abs(tstat);
    if tstat >= 2.326
        fprintf(fid,'%s: %7.4f*** (%6.4f)\n',coeff{ip},b(ip),se(ip));
    elseif tstat >= 1.96
        fprintf(fid,'%s: %7.4f** (%6.4f)\n',coeff{ip},b(ip),se(ip));
    elseif tstat >= 1.645
        fprintf(fid,'%s: %7.4f* (%6.4f)\n',coeff{ip},b(ip),se(ip));
    else
        fprintf(fid,'%s: %7.4f (%6.4f)\n',coeff{ip},b(ip),se(ip));
    end
end
fclose(fid);