%%% DERIVATIVES Coursework, Cass Business School, Msc Quantitative Finance - Alberto Ciampini%%%

%% Download data, source: Morningstar, period: 11/16/2011 to 11/16/2016 %%

GetData_Apple
GetData_IBM

%% Create time series objects and plot price time series %%

Datec=cellstr(Date);
ts1=fints(Datec,IBM,'IBM','daily');
ts2=fints(Datec,AAPL,'Apple','daily');

ts12=merge(ts1,ts2); 

%Plot
%figure;
%subplot(2,2,1);
%plot(ts1,'r'); %Stock 1 plot
%grid on;
%title('IBM daily stock prices');

%subplot(2,2,2);
%plot(ts2,'b'); %Stock 2 plot
%grid on;
%title('Apple daily stock prices');

%subplot(2,2,[3,4]);

plot(ts12);
grid on;
title('IBM vs Apple stock prices');
print -depsc Figure1

%% Briefly check the assumption of normality for log daily stock returns %%

%Create log periodic (daily) returns
rts1=diff(log(ts1));
rts2=diff(log(ts2));

Mrts1=fts2mat(rts1);
Mrts2=fts2mat(rts2);

figure;
subplot(2,1,1);
%Histogram of 1st stock log daily returns
nbins=100; %number of classes
h=histfit(Mrts1,nbins,'normal');
h(1).FaceColor=[[1 0 0]];
h(1).EdgeColor='w';
h(2).Color=[[1 1 0]];
title('Histogram of IBM daily returns');

subplot(2,1,2);
%Histogram of 2nd stock log daily returns with fitted normal lines
i=histfit(Mrts2,nbins,'normal');
i(1).FaceColor=[[0 0 1]];
i(1).EdgeColor='w';
i(2).Color=[[1 1 0]];
title('Histogram of Apple daily returns');
print -depsc Figure2

%Perform JB tests on daily log returns
alpha=0.05;
[Ho,p]=jbtest(Mrts1,alpha);
[H1,p1]=jbtest(Mrts2,alpha);
a=[Ho;H1];
p=[p;p1];
alpha1=[alpha;alpha];
comment=strcat({'are IBM daily log returns normally distributed?';...
                  'are Apple daily log returns normally distributed?'});
table(comment,a,alpha1,p) % 1: Ho rejected | 0: Ho accepted

%% Brief descriptive statistics %%

equities=strcat({'IBM log returns';'Apple log returns'});
daily_mean=[nanmean(rts1);nanmean(rts2)];
daily_std=[nanstd(rts1);nanstd(rts2)];
daily_vol=daily_std.^2;
skew=[skewness(Mrts1);skewness(Mrts2)]; 
kurt=[kurtosis(Mrts1);kurtosis(Mrts2)]; 

%Annualisation of returns moments
ndays=length(rts1);
ann=ndays/365;
ann_mean=daily_mean*sqrt(ann);
ann_std=daily_std*sqrt(ann);
ann_vol=ann_std.^2;

table(equities,daily_mean,ann_mean,daily_std,ann_std,daily_vol,ann_vol,skew,kurt)

%% Preliminary setup of MCSs %%

%% Step 1: calculate the B.S. option premium at t=0 from the spot price of 1st stock %%

S=fts2mat(ts1.IBM(ndays+1)); %extract the last price S (ndays: last price)
K=100; %strike price
sig=ann_std(1); %annualised standard deviation of S
r=0.03; %risk free rate
T=1; %time to maturity (years)

C=BS_European_Call(S,K,sig,r,T)

%% Step 2: simulate the change in option premium (Econometrics toolbox required) %%

%Specify a ONE DIMENSION RANDOM WALK (without drift) as data generating process
N=1; %dimension
F=@(t,X) zeros(N,1); %drift part (set to zero)
G=@(t,X) eye(N); %diffusion part (set to one)
%create a sde object (stochastic differential equation)
%NOTE: STARTING VALUE OF STOCK PRICE(S) SET TO THE LAST OBSERVED PRICE
S_MCS=sde(F,G,'StartState',S) 
                              
%Simulate 5 steps/innovations of this process
horizon=9;  %forecast horizon (according to the frequency of data)  
%X=S_MCS.simByEuler(horizon,'ntrials',1000,'Z',@(t,X) RandDir(N));
%ntrials: number of simulations, set to 1000                                                
%(Z: Specification of the dependent random noise process used to generate
%   the Brownian motion vector that drives the simulation)
X=S_MCS.simByEuler(horizon,'ntrials',1000); 

%Plot the simulated stock prices
plot(1:horizon+1,X(:,:))
ylabel('Simulated stock price');
xlabel('Time (trading days)');
title('Simulated IBM stock prices');
print -depsc Figure3

% figure('color','w');
% axes; 
% grid on;
% shg
% for i=1:length(X)
%     line(1:1:horizon+1,X(:,:,i),'color',rand(1,3));
%     drawnow
% end
% ylabel('Simulated stock price (IBM)');
% xlabel('Time (trading days)');

%Extract each simulated final price (at t=5 in this case)
%NOTE: start and end date of forecast horizon, FORMAT YYYY,MM,DD
Date(length(Date));
t1=datetime(2016,11,16); 
t2=datetime(2016,11,25); %10 non-trading days forecast
t=transpose(t1:t2);

comment=strcat({'daily stock prices forecast starting from'});
comment1=strcat({'up to'});
start_date=t1;
end_date=t(length(t));
simulation_horizon=length(t)
table(simulation_horizon,comment,start_date,comment1,end_date)

T1=T-((horizon+1)/252);
%Difference operator for the change in call premium
g=@(S,K,sig,r,T1) BS_European_Call(S,K,sig,r,T1)-C; 

%Simulate the CHANGE in call premium from the simulated stock price
for i=1:length(X)
    j=X(:,:,i);
    S_mcs=j(10);
    C_msc=g(S_mcs,K,sig,r,T1);
    A(:,i)=C_msc;
end

%Scatterplot of simulated final (t=10) IBM stock price (x) and relative 
%   change in call premium (y) 

for i=1:length(X)
    j=X(:,:,i);
    S_mcs=j(10);
    P(:,i)=S_mcs;
end

scatterhist(P,A,'Color','r');
xlabel('Simulated final stock price (IBM)');
ylabel('Simulated change in call premium');
title('Scatterplot of simulated changes in IBM-call premium as function of its synthetic final stock prices');
print -depsc Figure4

figure;
subplot(2,1,1);
%Histogram of simulated changes in call premiumn with fitted normal lines
nbins=100; %number of classes
h=histfit(A,nbins,'normal');
h(1).FaceColor=[[1 0 0]];
h(1).EdgeColor='w';
h(2).Color=[[1 1 0]];
title('Histogram of simulated changes in IBM-call premium');

subplot(2,1,2);
%Histogram of simulated IBM stock prices with fitted normal lines
i=histfit(X(1:length(A)),nbins,'normal');
i(1).FaceColor=[[0 0 1]];
i(1).EdgeColor='w';
i(2).Color=[[1 1 0]];
title('Histogram of IBM simulated final stock prices');
print -depsc Figure5

%% Step 3: Calculate 5% Var %%

%NOTE: number of held stocks and precision of the VaR (%) to be specified
n1=1000;
VaRPrec=0.05;
A_sort=sort(A);
n=VaRPrec*(length(A));
VaR=abs(n1*A_sort(n));

comment=strcat({'->'});
comment1=strcat({'-day VaR for a portfolio of'});
comment2=strcat({'European call options on IBM at a confidence interval of'});
table(VaR,comment,simulation_horizon,comment1,comment2,VaRPrec)

%% PORTFOLIO OF CALL OPTIONS %%

%% Step 1: calculate the B.S. option premium at t=0 from the spot price of 2nd stock %%

S1=fts2mat(ts2.Apple(ndays+1)); %extract the last price S1 (ndays: last price)
K=100; %strike price
sigA=ann_std(2); %annualised standard deviation of S1
r=0.03; %risk free rate
T=1; %time to maturity (years)

C1=BS_European_Call(S1,K,sigA,r,T);

%INITIAL VALUE OF PORTFOLIO OF 2 LONG CALLS: THE MINUS SIGN INDICATES THAT
%   WE HAVE SOLD/WRITTEN A CALL ON S1 (LONG CALL ON S INSTEAD)
Cp=C-C1;
comment=strcat({'Initial portfolio value'});
table(Cp,comment)

%% Step 2: simulate the change in option premium (Econometrics toolbox required) %%

rts12=merge(rts1,rts2);
rts12=fts2mat(rts12);

expReturn=diag(mean(rts12));  %expected return vector
sigma=diag(std(rts12));   %volatility of returns

%NOTE: WE NEED TO AMEND THE DATA GENERATING PROCESS and adjust for
%   (historical) correlation. Specify Correlation as the sample correlation
%   matrix of those returns. In this case, the components of the Brownian
%   motion are dependent:
correlation=corrcoef(rts12);
Start=[S; S1];
GBM1=gbm(expReturn,sigma,'Correlation',correlation,'StartState',Start);

rng(22814,'twister')
[X1,Tx1]=simByEuler(GBM1,horizon,'ntrials',1000);  %correlated Brownian motion

%Plot the simulated stock prices
plot(Tx1,X1(:,:))
ylabel('Simulated final stock price (Apple and IBM)');
xlabel('Time (trading day)');
print -depsc Figure6

%figure('color','w');
%axes; 
%grid on;
%shg
%for i=1:length(X1)
%    line(1:1:horizon+1,X1(:,:,i),'color',rand(1,3));
%    drawnow
%end
%ylabel('Simulated stock prices (Apple(l) and IBM(u))');
%xlabel('Time (trading days)');

T1=T-((horizon+1)/252);
%Difference operator for the change in call premium
g=@(S,K,sig,r,T1) BS_European_Call(S,K,sig,r,T1)-C; %Delta IBM call

f=@(S,K,sig,r,T1) BS_European_Call(S,K,sigA,r,T1)-C1; %Delta Apple call

%Simulate the CHANGE in call premia from the simulated stock prices

%1st stock change in call premium, FROM SECOND MCS (dependent BMs)
%IBM in this case (first column of X1(:,:,i))

for i=1:length(X1)
    j=X1(:,:,i);
    S_mcs=j(:,1);
    S_mcs1=S_mcs(10);
    C_msc=g(S_mcs1,K,sig,r,T1);
    A1(:,i)=C_msc;
end

%2nd stock change in call premium, FROM SECOND MCS (dependent BMs)
%Apple in this case (second column of X1(:,:,i))

for i=1:length(X1)
    j=X1(:,:,i);
    S_mcs=j(:,2);
    S_mcs2=S_mcs(10);
    C_mcs=f(S_mcs2,K,sigA,r,T1);
    A2(:,i)=C_mcs;
end

%Scatterplot of simulated final (t=10) IBM stock price (x) and relative 
%   change in call premium (y)  

for i=1:length(X1)
    j=X1(:,:,i);
    S_mcs=j(:,1);
    S_mcs1=S_mcs(10);
    P1(:,i)=S_mcs1;
end

scatterhist(P1,A1,'Color','r');
xlabel('Simulated final stock price (IBM)');
ylabel('Simulated change in call premium');
print -depsc Figure7

%Scatterplot of simulated final (t=10) Apple stock price (x) and relative 
%   change in call premium (y)  

for i=1:length(X1)
    j=X1(:,:,i);
    S_mcs=j(:,2);
    S_mcs2=S_mcs(10);
    P2(:,i)=S_mcs2;
end

scatterhist(P2,A2,'Color','b');
xlabel('Simulated final stock price (Apple)');
ylabel('Simulated change in call premium');
print -depsc Figure8

figure;
subplot(2,1,1);
%Histogram of simulated changes in IBM call premiumn with fitted normal lines
nbins=100; %number of classes
h=histfit(A1,nbins,'normal');
h(1).FaceColor=[[1 0 0]];
h(1).EdgeColor='w';
h(2).Color=[[1 1 0]];
title('Histogram of simulated changes in IBM call premium');

subplot(2,1,2);
%Histogram of simulated changes in Apple call premiumn with fitted normal lines
i=histfit(A2,nbins,'normal');
i(1).FaceColor=[[0 0 1]];
i(1).EdgeColor='w';
i(2).Color=[[1 1 0]];
title('Histogram of simulated changes in Apple call premium');
print -depsc Figure9

%% Step 3: Calculate 5% Var %%

%In dealing with a portfolio of options (long call on IBM and short call on
%   Apple), we need to calculate the change in the portfolio value (simulations
%   at t=10 minus initial portfolio value) for each simulation. Remember moreover that: 
%   Cp=C-C1;

DeltaPort_final=A1-A2; %minus sign to indicate a short position on 
%   Apple-call: if the price of apple increases I loose money as writer of the option
dV=DeltaPort_final-Cp;

%Histogram of simulated changes in portfolio value with fitted normal lines
i=histfit(dV,nbins,'normal');
i(1).FaceColor=[[0 1 1]];
i(1).EdgeColor='b';
i(2).Color=[[1 0 0]];
title('Histogram of simulated changes in portfolio value');
print -depsc Figure10

%NOTE: number of held stocks and precision of the VaR (%) to be specified
n1=1000;
VaRPrec=0.05;
dV_sort=sort(dV);
n=VaRPrec*(length(dV));
VaR=abs(n1*dV_sort(n));

comment=strcat({'->'});
comment1=strcat({'-day VaR for a portfolio of'});
comment2=strcat({'European call options: 1000 long IBM-call and 1000 short Apple-call, confidence interval of'});
table(VaR,comment,simulation_horizon,comment1,n1,comment2,VaRPrec)

%Perform JB tests 
alpha=0.05;
[Ho,p]=jbtest(A1,alpha);
[H1,p1]=jbtest(A2,alpha);
a=[Ho;H1];
p=[p;p1];
alpha1=[alpha;alpha];
comment=strcat({'are IBM changes in call premium normally distributed?';...
                  'are Apple changes in call premium normally distributed?'});
table(comment,a,alpha1,p) % 1: Ho rejected | 0: Ho accepted