%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                               %
% This is a demo for the JMVFG algorithm, which is proposed in the paper below. %
%                                                                               %
% Si-Guo Fang, Dong Huang, Chang-Dong Wang, Yong Tang.                          %
% Joint Multi-view Unsupervised Feature Selection and Graph Learning.           %
% IEEE Transactions on Emerging Topics in Computational Intelligence, 2023.     %
%                                                                               %
% The code has been tested in Matlab R2019b on a PC with Windows 10.            %
%                                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function demo_1_multiview_clustering()

clc;clear;close all;

dataName = 'HW';
load(['data_',dataName,'.mat'],'X','Y');
eta = 1;gamma = 1;beta = 1e-3; 

c = length(unique(Y));
tic;
[ranking,SS,~] = JMVFG(X,eta,gamma,beta,c);
toc;

% Use the learned similarity matrix SS to test the clustering task

fprintf('-------------------------------------------------------------\n')
fprintf('Multi-view clustering: \n')
[Label, ~] = SpectralClustering(SS,c);
Metric = NMImax(Label , Y);
fprintf('NMI = %f\n',Metric)


