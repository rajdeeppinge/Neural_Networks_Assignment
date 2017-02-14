%D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\approximationproblems

% Program for  RBF..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 8\ae.tra');
[TD,in] = size(Ntrain);

%Load testing data
NFeature=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 8\ae.tes');
[NTestD,~]=size(NFeature);

NAns=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Results\Group 8\ae.cla');


% Initialize the Algorithm Parameters.....................................
inp = in-1;          % No. of input neurons
K = 22;        % No. of hidden neurons
%out = 3;            % No. of Output Neurons
%lam = 1.e-03;       % Learning rate
%epo = 100;

trueOut = Ntrain(:, inp+1:end);

%find out the number of classes
Nclasses = size(unique(trueOut, 'rows'), 1);
out = Nclasses;            % No. of Output Neurons

% create the output vectors for the true/actual outputs for training data
Ytrue = zeros(TD, Nclasses);
for i = 1 : TD
   Ytrue(i, :) = -1;
   % can take another loop to traverse column-wise
   Ytrue(i, trueOut(i, 1)) = 1;     %%%%%%%%%%%%%%%CHECK matrix true value how to convert ?????
end

% create the output vectors for the true/actual outputs for testing data
YAns = zeros(NTestD, Nclasses);
for i = 1 : NTestD
   YAns(i, :) = -1;
   % can take another loop to traverse column-wise
   YAns(i, NAns(i, 1)) = 1;     %%%%%%%%%%%%%%%CHECK matrix true value how to convert ?????
end

%%%CROSS VALIDATION INITIALIZATION

% cross validation factor = 0.9
CVFactor = 0.9;
NTD = floor(TD * CVFactor);     %training data after cross validation

NCV = TD - NTD;     %cross validation testing sample

resultcheck_tra = zeros(min(TD,100)+1-(inp+out+1), 2);
resultcheck_tes = zeros(min(TD,100)+1-(inp+out+1), 2);

%for K = (inp + out + 1) : min(TD, 100)

    % random centres u
    %u = zeros(K, inp);
    k = randperm(TD);
    u = Ntrain(k(1:K),1:inp);

    xinp = Ntrain(:, 1:inp);
    %for i = 1 : K
    %   u(i, :) = xinp(floor(rand()*NTestD)+1, :);
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
    G = zeros(TD, K);

    for i = 1 : TD
       %for j = 1 : K
       %    G(i,j) = exp( - sum( (xinp(i, :) - u(j, :)).^2 ) / (2 * sigma * sigma) );
       %end

       G(i,:) = exp( - sum ( (( repmat(xinp(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 

    end

    %trainOut = Ntrain(:,inp+1:end);

    % find weights
    Weight = pinv(G) * Ytrue;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % test over training data
    conf_mat_tra = zeros(out, out);

    %rmstra = zeros(out,1);
    %res_tra = zeros(TD,2*out);

    %output vector d
    %d = zeros(TD, 1);



    %for sa = 1: NTestD
    %        xx = NFeature(sa,1:inp)';   % Current Sample
    %        ca = NFeature(sa,end);      % Actual Output

    %        tt = Ytrue(sa,:)';

    %        for neu = 1 : K
    %            d(sa) = d(sa) + Weight(neu) * exp( - sum( (xx' - u(neu, :)).^2 ) / (2 * sigma * sigma) );
    %        end

    %        rmstes = rmstes + (ca-d(sa)).^2;
    %        res_tes(sa,:) = [ca d(sa)];
    %end
    %disp('testing')
    %disp(sqrt(rmstes/NTestD))


    % inter matrix
    G = zeros(TD, K);

    for i = 1 : TD
       %for j = 1 : K
       %    G(i,j) = exp( - sum( (xinp(i, :) - u(j, :)).^2 ) / (2 * sigma * sigma) );
       %end

       G(i,:) = exp( - sum ( (( repmat(xinp(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 

    end

    output_tra = G * Weight;


    conf_mat_tra = zeros(out,out);

    [~,pred_label_tra]= max(output_tra,[],2); % the class is taken as max prob in the coded vec


    % build the confusion matrix
    for sa=1:size(pred_label_tra)
        conf_mat_tra(trueOut(sa),pred_label_tra(sa))=conf_mat_tra(trueOut(sa),pred_label_tra(sa))+1;
    end

    %disp(conf_mat_tra) % we have obtained the confusion matrix

    
    predict_tra = [trueOut pred_label_tra];


    %correct classifications
    correct_tra = sum(diag(conf_mat_tra));

    %overall accuracy
    overall_acc_tra = 100*correct_tra/TD;

    %to get average accuracy

    classLabel = unique(trueOut, 'rows');

    % using histogram to get frequency of labels
    [labelCount_tra, classLabel] = hist(Ntrain(:,end), unique(classLabel));

    %average accuracy
    avg_acc_tra = 100/out * sum(diag(conf_mat_tra)./labelCount_tra');

    %geometric-mean accuracy
    geo_mean_acc_tra = nthroot(prod(100*diag(conf_mat_tra)./labelCount_tra'),out);

    
    resultcheck_tra(K-(inp+out), :) = [K overall_acc_tra];

    


    % Test the network.........................................................



    %rmstes = zeros(out,1);
    %res_tes = zeros(NTestD,2);

    %output vector d
    %d = zeros(NTestD, 1);

    %for sa = 1: NTestD
    %        xx = NFeature(sa,1:inp)';   % Current Sample
    %        ca = NFeature(sa,end);      % Actual Output

    %        for neu = 1 : K
    %            d(sa) = d(sa) + Weight(neu) * exp( - sum( (xx' - u(neu, :)).^2 ) / (2 * sigma * sigma) );
    %        end

    %        rmstes = rmstes + (ca-d(sa)).^2;
    %        res_tes(sa,:) = [ca d(sa)];
    %end
    %disp('testing')
    %disp(sqrt(rmstes/NTestD))

    % inter matrix
    G_tes = zeros(NTestD, K);

    for i = 1 : NTestD
       %for j = 1 : K
       %    G(i,j) = exp( - sum( (xinp(i, :) - u(j, :)).^2 ) / (2 * sigma * sigma) );
       %end

       G_tes(i,:) = exp( - sum ( (( repmat(NFeature(i,:),K,1) ) - u(:,:)).^2, 2 ) / (2*sigma*sigma) ); 

    end

    output_tes = G_tes * Weight;

    conf_mat_tes = zeros(out, out);

    [~,pred_label_tes]= max(output_tes,[],2); % the class is taken as max prob in the coded vec


    % build the confusion matrix from pred_lab and actual_lab
    for sa=1:size(pred_label_tes)
     conf_mat_tes(NAns(sa),pred_label_tes(sa))=conf_mat_tes(NAns(sa),pred_label_tes(sa))+1;
    end
    
    predict_tes = [NAns pred_label_tes];

    %disp(conf_mat_tes) % we have obtained the confusion matrix

    %correct classifications
    correct_tes = sum(diag(conf_mat_tes));

    %overall accuracy
    overall_acc_tes = 100*correct_tes/NTestD;


    %to get average accuracy

    % using histogram to get frequency of labels
    [labelCount_tes,classLabel] = hist(NAns, unique(classLabel));

    %average accuracy
    avg_acc_tes = 100/out * sum(diag(conf_mat_tes)./labelCount_tes');

    %geometric-mean accuracy
    geo_mean_acc_tes = nthroot(prod(100*diag(conf_mat_tes)./labelCount_tes'),out);

    resultcheck_tes(K-(inp+out), :) = [K overall_acc_tes];

       
%end

%plot(resultcheck_tra(:, 1), resultcheck_tra(:, 2))
%hold on
%plot(resultcheck_tes(:, 1), resultcheck_tes(:, 2), '-r')
%xlabel('No. of Hidden Neurons')
%ylabel('Overall Accuracy')
%Title('Graph to find optimum neurons based on results')
%legend('Training Data','Testing Data','Location','northwest')
%axis([10,70,85,105])