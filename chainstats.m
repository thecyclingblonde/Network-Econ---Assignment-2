function y=chainstats(chain,fid)
%CHAINSTATS some statistics from the MCMC chain
% chainstats(chain)
%    chain    nsimu*npar MCMC chain

% $Revision: 1.4 $  $Date: 2009/08/13 15:47:35 $

if nargin<3, fid=1; end % fid=1, standard output

names = {'alpha' 'beta' 'gamma' 'rho' 'eta' 'lambda' 'tau' 'b' 'c' 'xi' 'chi' 'vartheta' 'kappa'};

mcerr = bmstd(chain)./sqrt(size(chain,1));

[z,p]  = geweke(chain);
tau    = iact(chain);
stats  = [mean(chain)',std(chain)',mcerr',tau', p'];

[m,n] = size(stats);

fprintf(fid,'MCMC statistics, nsimu = %g \n\n', size(chain,1));

if nargin>1
  fprintf(fid,'% 10s ','');
end
fprintf(fid,'% 10s  % 10s  % 10s  % 10s  % 10s\n','mean','std','MC_err','tau','geweke');
if nargin>1
  fprintf(fid,'-----------');
end
fprintf(fid,'----------------------------------------------------------\n');
for i = 1:m
  if nargin>1
    fprintf(fid,'% 10s ',names{i});
  end
  fprintf(fid,'%10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n',stats(i,:));
end
if nargin>1
  fprintf(fid,'-----------');
end
fprintf(fid,'----------------------------------------------------------\n');
fprintf(fid,'\n');

if nargout>0
  %  y=[stats,stats2];
  y=stats;
end

return