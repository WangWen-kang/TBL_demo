%% LOAD DATA
% Turbulent Boundary layer data from Univeristy of Melbourne
% https://figshare.unimelb.edu.au/articles/dataset/Two-point_high_Reynolds_number_zero-pressure_gradient_turbulent_boundary_layer_dataset/12101088
u_data_c1=load('.\data\Inner_outer_u_z32_c1.mat');
u_data_c2=load('.\data\Inner_outer_u_z32_c2.mat');
u_data_c3=load('.\data\Inner_outer_u_z32_c3.mat');
load('.\data\Inner_outer_tbl_param.mat');

data_c1 = u_data_c1.data(1:25:end,:);data_c1=(data_c1-mean(data_c1,1))./std(data_c1,1);
data_c2 = u_data_c2.data(1:25:end,:);data_c2=(data_c2-mean(data_c2,1))./std(data_c2,1);
data_c3 = u_data_c3.data(1:25:end,:);data_c3=(data_c3-mean(data_c3,1))./std(data_c3,1);
%% Visualize DATA
set(0,'defaultaxesfontsize',14);
set(0,'defaulttextfontsize',14);
ts=(0:length(data_c1)-1)/ fs * utau^2 / nu*25;
figure('color','w'),
subplot(2,1,1),plot(ts,data_c1(:,1));set(gca,'xlim',[0,20000],'linewidth',1.5)
ylabel('$u_O/\sigma_u$','interpreter','latex')
subplot(2,1,2),plot(ts,data_c1(:,2));set(gca,'xlim',[0,20000],'linewidth',1.5)
xlabel('$t^+$','interpreter','latex')
ylabel('$u_I/\sigma_u$','interpreter','latex')
%% GC test with MVGC
% The Multivariate Granger Causality (MVGC) Toolbox
% https://github.com/lcbarnett/MVGC1
% Lionel Barnett and Anil K. Seth, "The MVGC Multivariate Granger Causality Toolbox: A new approach to Granger-causal inference", J. Neurosci. Methods 223 (2014), pp 50-68. DOI: 10.1016/j.jneumeth.2013.10.018

ntrials   = 3;     % number of trials
nobs      = 90000;   % number of observations per trial
regmode   = 'LWR';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
icregmode = 'LWR';  % information criteria regression mode ('OLS', 'LWR' or empty for default)
morder    = 'AIC';  % model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
momax     = 200;     % maximum model order for model order estimation
acmaxlags = 400;   % maximum autocovariance lags (empty for automatic calculation)
tstat     = '';     % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
alpha     = 0.05;   % significance level for significance test
mhtc      = 'FDR';  % multiple hypothesis test correction (see routine 'significance')

X=cat(3,transpose(data_c1),transpose(data_c2),transpose(data_c3));

ptic('\n*** tsdata_to_infocrit\n');
[AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
ptoc('*** tsdata_to_infocrit took ');
% Plot information criteria.

figure(1); clf;
plot_tsdata([AIC BIC]',{'AIC','BIC'},1/fs);
title('Model order estimation');

morder = moAIC;

ptic('\n*** tsdata_to_var... ');
[A,SIG] = tsdata_to_var(X,morder,regmode);
ptoc;
assert(~isbad(A),'VAR estimation failed');

ptic('*** var_to_autocov... ');
[G,info] = var_to_autocov(A,SIG,acmaxlags);
ptoc;
var_acinfo(info,true); % report results (and bail out on error)

ptic('*** autocov_to_pwcgc... ');
F = autocov_to_pwcgc(G);
ptoc;

% Check for failed GC calculation

assert(~isbad(F,false),'GC calculation failed');

% Significance test using theoretical null distribution, adjusting for multiple
% hypotheses.

pval = mvgc_pval(F,morder,nobs,ntrials,1,1,0,tstat); % take careful note of arguments!
sig  = significance(pval,alpha,mhtc);

% Plot time-domain causal graph, p-values and significance.

figure(2); clf;
sgtitlex('Pairwise-conditional Granger causality - time domain');
subplot(1,3,1);
plot_pw(F);
title('Pairwise-conditional GC');
subplot(1,3,2);
plot_pw(pval);
title('p-values');
subplot(1,3,3);
plot_pw(sig);
title(['Significant at p = ' num2str(alpha)])
%% TE with JIDT
% Java Information Dynamics Toolkit (JIDT)
% https://github.com/jlizier/jidt
% Joseph T. Lizier, "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems", Frontiers in Robotics and AI 1:11, 2014; doi:10.3389/frobt.2014.00011 (pre-print: arXiv:1408.3270)

% Add JIDT jar library to the path, and disable warnings that it's already there:
warning('off','MATLAB:Java:DuplicateClass');
javaaddpath('D:\wwk_tool\infodynamics-dist-1.5\infodynamics.jar');
% Add utilities to the path
addpath('D:\wwk_tool\infodynamics-dist-1.5\demos\octave');
NN=10;
% 0. Load/prepare the data:
% 1. Construct the calculator:
calc = javaObject('infodynamics.measures.discrete.TransferEntropyCalculatorDiscrete',NN, 1, 1, 1, 1, 1);
% 2. No other properties to set for discrete calculators.
mUtils = javaObject('infodynamics.utils.MatrixUtils');
sym={'uO','uI'};
% Compute for all pairs:
for s = 1:2
	for d = 1:2
		% For each source-dest pair:
		% Column indices start from 1 in Matlab:
		source = mUtils.discretise(octaveToJavaDoubleArray(data_c1(:,s)), NN);
		destination = mUtils.discretise(octaveToJavaDoubleArray(data_c1(:,d)), NN);

		% 3. Initialise the calculator for (re-)use:
		calc.initialise();
		% 4. Supply the sample data:
		calc.addObservations(source, destination);
		% 5. Compute the estimate:
		result = calc.computeAverageLocalOfObservations();

		fprintf('TE_Binned(%s -> %s) = %.4f bits\n', ...
			sym{s},sym{d}, result);
	end
end
%% CCM with CCM_L_M
% convergent cross mapping matlab toolbox
% https://ww2.mathworks.cn/matlabcentral/fileexchange/52964-convergent-cross-mapping
% Krakovsk¨¢, Anna, et al. "Causality studied in reconstructed state space. Examples of uni-directionally connected chaotic systems." arXiv preprint arXiv:1511.00505 (2015).

% parameters
tau = 1; % time step 
E   = 4; % dimension of reconstruction
LMN = 5; % number of neigborhoods for L and M methods
[ SugiC , SugiR , LM , SugiY , SugiX , origY , origX ]=SugiLM(data_c1(1:1:1000,1),data_c1(1:1:1000,2),tau,E,LMN);        

% results
disp('Sugiharas CMM correlation for estimate of X and original X in coupled case')
SugiC(1)
disp('Sugiharas CMM correlation for estimate of Y and original Y in coupled case')
SugiC(2)
figure;
plot(SugiY,origY,'ro',SugiX,origX,'b*')
title('Estimated vs. original data in coupled case')
xlabel('Estimated data') 
ylabel('Original data') 
legend('Y','X')

L=100:100:1000; % vector of computing points

CCM(data_c1(1:1:1000,1),data_c1(1:1:1000,2),tau,E,LMN,L);

%% DBN with GLOBALMIT
%GlobalMIT toolbox
%https://ww2.mathworks.cn/matlabcentral/fileexchange/32428-globalmit-toolbox
%Vinh, N. X., Chetty, M., Coppel, R., and Wangikar, P. P. (2011). GlobalMIT: Learning Globally Optimal Dynamic Bayesian Network with the Mutual Information Test (MIT) Criterion, Bioinformatics, in press.

n_state=10;

len=90000;
start=5000;
lag=-10;

data_c1_out=data_c1(start+lag:start+len+lag,1);
data_c1_in=data_c1(start:start+len,2);
data_c1_lag=cat(2,data_c1_out,data_c1_in);
data_c2_out=data_c2(start+lag:start+len+lag,1);
data_c2_in=data_c2(start:start+len,2);
data_c2_lag=cat(2,data_c2_out,data_c2_in);
data_c3_out=data_c3(start+lag:start+len+lag,1);
data_c3_in=data_c3(start:start+len,2);
data_c3_lag=cat(2,data_c3_out,data_c3_in);

a1= myIntervalDiscretize(data_c1_lag,n_state);
a2= myIntervalDiscretize(data_c2_lag,n_state);
a3= myIntervalDiscretize(data_c3_lag,n_state);

[b,c]=multi_time_series_cat(a1,a2,a3);
alpha=0.99;
[best_net_ab,score]=globalMIT_ab(b,c,alpha,1);
createDotGraphic(best_net_ab,{'uO','uI'});

%% SURD 
%please refer to "https://github.com/Computational-Turbulence-Group/SURD/blob/main/examples/E10_inner_outer.ipynb"