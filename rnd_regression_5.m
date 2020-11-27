clear;
load ./Data/data;

L=2;     % number of Monte Carlo repetition
N=100;   % number of nodes in the network
T=3000; % length of MCMC
R=2;     % number of iterations for simulating the network

gamma_T=zeros(7,T);
lambda_T=zeros(T,1);
beta_T=zeros(T,1);

for l=1:L % Monte Carlo repetitions
    
    W=WW{l};
    C=CC{l};
    X=XX{l};
    Y=YY{l};
    
    % jumping rate in proposal distributions
    
    c_1=1e-5;
    c_2=1e-4;
    c_3=1e-2;
    acc_1=0.0;
    acc_rate1=zeros(T,1);
        
    % initial values to start MCMC 
    
    gamma_T(:,1)=[-3.0, 1.0, -0.10, 0.3, -0.03, 0.5, 0.60 ];
    lambda_T(1)=0.0100;
    beta_T(1)= 0.8000;
    
    % hyper parameters 
    
    beta_0=0;
    gamma_0=zeros(1,7);
    lambda_0=0.0;
    G_0=eye(7)*100.0;
    B_0=100.0;
    
    
    for t=2:T  % start MCMC
        tic;
        
        % propose gamma by Adaptive M-H (Haario, H., Saksman, E., Tamminen, J.: An adaptive Metropolis algorithm. Bernoulli 7(2), 223-242 (2001))
        accept=0;
        while accept==0
            if t<500
                gamma_1=mvnrnd(gamma_T(:,t-1)',eye(7)*c_1);
            else
                gamma_1=mvnrnd(gamma_T(:,t-1)',cov(gamma_T(:,1:t-1)')*2.38^2/7)*0.6+mvnrnd(gamma_T(:,t-1)',eye(7)*c_1)*0.4;
            end
            if  gamma_1(7)>0
                accept=1;
            end
        end
        gamma_2=gamma_T(:,t-1);
        
        % propose lambda by Adaptive M-H%
        accept=0;
        while accept==0
            if t<=500
                lambda_1=randn(1)*c_2+lambda_T(t-1);
            else
                lambda_1=mvnrnd(lambda_T(t-1)',cov(lambda_T(1:t-1)')*2.38^2)*0.6+...
                    mvnrnd(lambda_T(t-1)',eye(1)*c_2^2)*0.4;
            end
            if abs(lambda_1)<=1/20
                accept=1;
            end
        end
        
        % propose beta by Adaptive M-H %
        if t<=500
            beta_1=randn(1)*c_3+beta_T(t-1);
        else
            beta_1=mvnrnd(beta_T(t-1)',cov(beta_T(1:t-1)')*2.38^2)*0.6+...
                mvnrnd(beta_T(t-1)',eye(1)*c_3^2)*0.4;
        end
        
        H=zeros(N,N);
        for i=1:N
            for j=1:N
                if j~=i
                    H(i,j)=gamma_1(1)+gamma_1(2)*C(i,j)+gamma_1(3)*abs(X(i)-X(j));
                end
            end
        end
        
        S=eye(N)-lambda_1*W;
        S_INV=inv(S);
        
        S_new=S;
        S_old=S;
        W_old=W;
        W_new=W;
        
        FE=X*beta_1;
        
        S_INV_OLD=S_INV;
        Y2_star=S\FE;
        S_INV_OLD2=S_INV_OLD*gamma(7);
        Y2=mvnrnd(zeros(N,1),S_INV_OLD2,1)'+Y2_star;
        
        loglike_y_old=log(mvnpdf(Y2,Y2_star,S_INV_OLD2));
        
        for r=1:R  % start to simulate auxiliary network and outcome   
            for i=1:N
                for j=1:N
                    if i~=j
                        W_new(i,j)=1-W_new(i,j);
                        W_new(j,i)=1-W_new(j,i);
                        
                        S_INV_TEMP=S_INV;
                        if W_new(i,j)==1
                            S_INV_TEMP=-(-lambda_1)/(1+(-lambda_1)*S_INV(i,j))*S_INV(1:N,i)*S_INV(j,1:N)+S_INV_TEMP;
                            S_INV_NEW=S_INV_TEMP;
                            S_INV_NEW=-(-lambda_1)/(1+(-lambda_1)*S_INV_TEMP(i,j))*S_INV_TEMP(1:N,j)*S_INV_TEMP(i,1:N)+S_INV_NEW;
                            S_new(i,j)=S_new(i,j)-lambda_1;
                            S_new(j,i)=S_new(j,i)-lambda_1;
                        else
                            S_INV_TEMP=(-lambda_1)/(1-(-lambda_1)*S_INV(i,j))*S_INV(1:N,i)*S_INV(j,1:N)+S_INV_TEMP;
                            S_INV_NEW=S_INV_TEMP;
                            S_INV_NEW=(-lambda_1)/(1-(-lambda_1)*S_INV_TEMP(i,j))*S_INV_TEMP(1:N,j)*S_INV_TEMP(i,1:N)+S_INV_NEW;
                            S_new(i,j)=S_new(i,j)+lambda_1;
                            S_new(j,i)=S_new(j,i)+lambda_1;
                        end
                        
                        loglike_y_new=loglike_y_old;
                        
                        if rand(1)<=0.01  % update outcome
                            Y1_star=S_INV_NEW\FE;
                            S_INV_NEW2=S_INV_NEW*gamma_1(7);
                            Y1=mvnrnd(zeros(N,1),S_INV_NEW2,1)'+Y1_star;
                            loglike_y_new=log(mvnpdf(Y1,Y1_star,S_INV_NEW2));
                        else
                            Y1=Y2;
                        end
                        
                        PHI1=FE'*Y1-0.5*Y1'*S_new*Y1;
                        PHI2=FE'*Y2-0.5*Y2'*S_old*Y2;
                        
                        popularity=sum(W_new(i,:))-W_new(i,j)+sum(W_new(j,:))-W_new(i,j);
                        congestion=popularity^2;
                        cyclic=W_new(i,:)*W_new(j,:)'-W_new(i,j);
                        
                        p_w=(H(i,j)+gamma_1(4)*popularity+gamma_1(5)*congestion+gamma_1(6)*cyclic)*(-1)^(1-W_new(i,j))+PHI1-PHI2;
                        
                        p_w=p_w/gamma_1(7)+loglike_y_new-loglike_y_old;
                        
                        if log(rand(1))<=p_w
                            W_old(i,j)=W_new(i,j);
                            W_old(j,i)=W_new(j,i);
                            S_old(i,j)=S_old(i,j);
                            S_old(j,i)=S_old(j,i);
                            S_INV=S_INV_NEW;
                            loglike_y_old=loglike_y_new;
                            Y2=Y1;
                        end
                        W_new(i,j)=W_old(i,j);
                        W_new(j,i)=W_old(j,i);
                        S_new(i,j)=S_old(i,j);
                        S_new(j,i)=S_old(j,i);
                    end
                end
            end
        end
        
        if (abs(sum(sum(W_new,2))-sum(sum(W,2)))>50)  % (arbitrary) condition to reject auxiliary network 
            gamma_T(:,t)=gamma_T(:,t-1);
            lambda_T(t)=lambda_T(t-1);
            beta_T(t)=beta_T(t-1);
        else
            psi_1=zeros(N,N);
            psi_2=zeros(N,N);
            psi_3=zeros(N,N);
            psi_4=zeros(N,N);
            for i=1:N
                for j=1:N
                    if j~=i
                        popularity=sum(W(i,:))-W(i,j)+sum(W(j,:))-W(i,j);
                        congestion=popularity^2;
                        cyclic=W(i,:)*W(j,:)'-W(i,j);
                        
                        popularity_new=sum(W_new(i,:))-W_new(i,j)+sum(W_new(j,:))-W_new(i,j);
                        congestion_new=popularity_new^2;
                        cyclic_new=W_new(i,:)*W_new(j,:)'-W_new(i,j);
                        
                        psi_1(i,j)=gamma_1(1)+gamma_1(2)*C(i,j)+gamma_1(3)*abs(X(i)-X(j)) ...
                            +gamma_1(4)*popularity+gamma_1(5)*congestion+(1.0/3.0)*gamma_1(6)*cyclic;
                        
                        psi_2(i,j)=gamma_2(1)+gamma_2(2)*C(i,j)+gamma_2(3)*abs(X(i)-X(j)) ...
                            +gamma_2(4)*popularity+gamma_2(5)*congestion+(1.0/3.0)*gamma_2(6)*cyclic;
                        
                        psi_3(i,j)=gamma_2(1)+gamma_2(2)*C(i,j)+gamma_2(3)*abs(X(i)-X(j)) ...
                            +gamma_2(4)*popularity_new+gamma_2(5)*congestion_new+(1.0/3.0)*gamma_2(6)*cyclic_new;
                        
                        psi_4(i,j)=gamma_1(1)+gamma_1(2)*C(i,j)+gamma_1(3)*abs(X(i)-X(j)) ...
                            +gamma_1(4)*popularity_new+gamma_1(5)*congestion_new+(1.0/3.0)*gamma_1(6)*cyclic_new;
                        
                    end
                end
            end
            
            psi_1=psi_1/gamma_1(7);
            psi_2=psi_2/gamma_T(7,t-1);
            psi_3=psi_3/gamma_T(7,t-1);
            psi_4=psi_4/gamma_1(7);
            
            p_w= trace(psi_1*W)-trace(psi_2*W)+trace(psi_3*W_new)-trace(psi_4*W_new);
            
            S1=eye(N)-lambda_1*W;
            S2=eye(N)-lambda_T(t-1)*W;
            
            FE1=X*beta_1;
            FE2=X*beta_T(t-1);
            
            PHI1=FE1'*Y-0.5*Y'*S1*Y;
            PHI2=FE2'*Y-0.5*Y'*S2*Y;
            
            PHI1=PHI1/gamma_1(7);
            PHI2=PHI2/gamma_T(7,t-1);
            
            S3=eye(N)-lambda_T(t-1)*W_new;
            S4=eye(N)-lambda_1*W_new;
            
            PHI3=FE2'*Y2-0.5*Y2'*S3*Y2;
            PHI4=FE1'*Y2-0.5*Y2'*S4*Y2;
            
            PHI3=PHI3/gamma_T(7,t-1);
            PHI4=PHI4/gamma_1(7);
            
            pp=p_w/2+(PHI1-PHI2)+(PHI3-PHI4);
            
            pp=pp+log(mvnpdf(gamma_1,gamma_0,G_0))-log(mvnpdf(gamma_T(:,t-1)',gamma_0,G_0))...
                +log(mvnpdf(beta_1,beta_0,B_0))-log(mvnpdf(beta_T(t-1),beta_0,B_0));
            
            if log(rand(1))<=pp
                gamma_T(:,t)=gamma_1;
                lambda_T(t)=lambda_1;
                beta_T(t)=beta_1;
                acc_1=acc_1+1.0;
            else
                gamma_T(:,t)=gamma_T(:,t-1);
                lambda_T(t)=lambda_T(t-1);
                beta_T(t)=beta_T(t-1);
            end
        end
        
        acc_rate1(t)=acc_1/t;
        time=toc;
        
        if (t/1)-round(t/1)==0
            fprintf('t=%d\n',t);
            fprintf('time= %5.3f Secs\n',time);
            fprintf('lambda= %5.3f\n',lambda_T(t));
            fprintf('beta= %5.3f\n',beta_T(t));
            fprintf('gamma= %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n',gamma_T(:,t)');
            fprintf('acc_rate1= %5.3f\n',acc_rate1(t));
            fprintf('\n');
        end      
    end
        
end

%% Plot MCMC draws.
figure();
set(gca,'Layer','top')
set(gca,'FontSize',18);
subplot(2,1,1)
set(gca,'defaulttextinterpreter','latex')
plot(lambda_T,'-or');
hold on 
plot(ones(length(lambda_T),1)*mean(lambda_T),'-k');
hold on
plot(ones(length(lambda_T),1)*mean(lambda_T)+std(lambda_T),'--k');
hold on
plot(ones(length(lambda_T),1)*mean(lambda_T)-std(lambda_T),'--k');
ylabel('$\lambda$')
xlabel('$t$')
subplot(2,1,2)
set(gca,'defaulttextinterpreter','latex')
plot(beta_T,'-ob');
hold on 
plot(ones(length(beta_T),1)*mean(beta_T),'-k');
hold on
plot(ones(length(beta_T),1)*mean(beta_T)+std(beta_T),'--k');
hold on
plot(ones(length(beta_T),1)*mean(beta_T)-std(beta_T),'--k');
ylabel('$\beta$')
xlabel('$t$')

%% Analyze convergence following Geweke (1992).
chain = horzcat(lambda_T,beta_T);
chainstats(chain)

%% Compute p-values under the assumption of asymptotic normality.
z = mean(chain)./std(chain);
pvalue = 2*(1 - normcdf(z));
disp(['P-values: ' num2str(pvalue)])
