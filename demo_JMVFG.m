function demo_JMVFG()

clc;clear;close all;

dataName = 'HW';
load(['data_',dataName,'.mat'],'X','Y');
eta = 1;gamma = 1;beta = 1e-3; 

c = length(unique(Y));
tic;
[ranking,SS,~] = JMVFG(X,eta,gamma,beta,c);
toc;

warning off
fprintf('-------------------------------------------------------------\n')
fprintf('Multi-view clustering: \n')
[Label, ~] = SpectralClustering(SS,c);
Metric = NMImax(Label , Y);
fprintf('NMI = %f\n',Metric)

