%D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\approximationproblems

% Program for  RBF..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\her.tra');
[NTD,~] = size(Ntrain);

% Initialize the Algorithm Parameters.....................................
inp = 2;          % No. of input neurons
K = 40;        % No. of hidden neurons
out = 1;            % No. of Output Neurons
%lam = 1.e-03;       % Learning rate
%epo = 100;

% random centres u
%u = zeros(K, inp);
k = randperm(NTD);
u = Ntrain(k(1:K),1:inp);

xinp = Ntrain(:, 1:inp);
%for i = 1 : K
%   u(i, :) = xinp(floor(rand()*NTD)+1, :);
%end

% setting spread sigma
dist = zeros(K, K);
for i = 1 : K
   for j = i+1 : K
      dist(i,j) = sqrt( sum((u(i, :) - u(j, :)).^2) );
      dist(j,i) = dist(i, j);
   end
end

dmax = max(dist(:));

sigma = dmax / sqrt(K);

% inter matrix
G = zeros(NTD, K);

for i = 1 : NTD
    %for j = 1 : K
   %    G(i,j) = exp( - sum( (xinp(i, :) - u(j, :)).^2 ) / (2 * sigma * sigma) );
   %end
    
   G(i,:) = exp( - sum ( (( repmat(xinp(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 
end

trainOut = Ntrain(:,inp+1:end);

% find weights
Weight = pinv(G) * trainOut;


% Test the network.........................................................
NFeature=load('her.tes');
[NTD,~]=size(NFeature);
rmstes = zeros(out,1);
res_tes = zeros(NTD,2);

%output vector d
d = zeros(NTD, 1);

for sa = 1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        ca = NFeature(sa,end);      % Actual Output
        
        for neu = 1 : K
            d(sa) = d(sa) + Weight(neu) * exp( - sum( (xx' - u(neu, :)).^2 ) / (2 * sigma * sigma) );
        end
        
        rmstes = rmstes + (ca-d(sa)).^2;
        res_tes(sa,:) = [ca d(sa)];
end
disp('testing')
disp(sqrt(rmstes/NTD))
