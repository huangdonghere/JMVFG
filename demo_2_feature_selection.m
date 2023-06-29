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

function demo_2_feature_selection()

clc;clear;close all;

dataName = 'HW';
load(['data_',dataName,'.mat'],'X','Y');
eta = 1;gamma = 1;beta = 1e-3; 

c = length(unique(Y));
tic;
[ranking,SS,XX] = JMVFG(X,eta,gamma,beta,c);
toc;

% Use the feature ranking vector to select features and test the feature
% selection task.

fprintf('-------------------------------------------------------------\n')
fprintf('Multi-view unsupervised feature selection: \n')
M = size(XX,2); %The number of total features.
prop = 0.1; %The proportion of selected features.
Xsub = XX(: , ranking(1 : floor(prop*M)));
Metric = zeros(20,1);
for i = 1:20
    Label = litekmeans(Xsub, c,'Replicates',5); %label by k-means
    Metric(i) = NMImax(Label , Y); 
end
fprintf('NMI(mean) = %f, NMI(std) = %f\n',mean(Metric(:)),std(Metric(:)))