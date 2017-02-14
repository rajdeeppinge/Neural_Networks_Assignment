% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc

% Load the training data..................................................
Ntrain=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 5\Iris.tra');
[TD,in] = size(Ntrain);      % TD means total data samples


% Initialize the Algorithm Parameters.....................................
inp = in - 1;    % No. of input neurons, 1 less because last column is the label
hid = 12;        % No. of hidden neurons
%out = 3;            % No. of Output Neurons
lam = 1.e-02;       % Learning rate
epo = 1000;

alpha = 0.5;     % momentum constant

trueOut = Ntrain(:, inp+1:end);

%find out the number of classes
Nclasses = size(unique(trueOut, 'rows'), 1);
out = Nclasses;            % No. of Output Neurons

% create the output vectors for the true/actual outputs
Ytrue = zeros(TD, Nclasses);
for i = 1 : TD
   Ytrue(i, :) = -1;
   % can take another loop to traverse column-wise
   Ytrue(i, trueOut(i, 1)) = 1;     %%%%%%%%%%%%%%%CHECK matrix true value how to convert ?????
end


%%%CROSS VALIDATION INITIALIZATION

% cross validation factor = 0.9
CVFactor = 0.9;
NTD = floor(TD * CVFactor);     %training data after cross validation

NCV = TD - NTD;     %cross validation testing sample

% split the data in proportion
%NcrossValData = Ntrain;
%[values, order] = sort(NcrossValData(:,end));
%NcrossValData = NcrossValData(order,:);

%classLabel = unique(trueOut, 'rows');

% using histogram to get frequency of labels
%[labelCount,classLabel] = hist(NcrossValData(:,end), unique(classLabel));

%crossValCount = floor(CVFactor * labelCount);

%CVtrainData = zeros(sum(crossValCount), inp);
%CVtestData = zeros(sum(labelCount-crossValCount), inp);

%for i = 1 : Nclasses
%    classData = NcrossValData(:, end==classLabel(i));
%    CVtrainData = labelCount(i)
%    CVtrainData(NcrossValData((i-1)*labelCount(i) ,:) )
%end


% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
traierrvec=zeros(epo,1);

for ep = 1 : epo
    correct = 0;
    sumerr = 0;
    
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        ttCur = Ntrain(sa,inp+1:end)'; % Current Target
        tt = Ytrue(sa, :)';          %coded output
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh') + alpha * DWoOld;                   % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx' + alpha * DWiOld;    %update for input weight
        % updating hidden weights
           % DWi = DWi + DeltaW
           % DeltaW = lam * deltaj * xij
           % deltaj = oj(here Yh) * (1-oj) * summation(Wojk * deltak)
           % deltak = er = tt - Yo
        
        sumerr = sumerr + sum(er.^2);
        
        [val, class] = max(Yo);
        
        if ttCur == class
            correct = correct + 1;
        end
    end
    traierrvec(ep)=sumerr/NTD;
%    disp([sumerr NTD-correct])
    
    Wi = Wi + DWi;
    Wo = Wo + DWo;
    
    DWiOld = DWi;
    DWoOld = DWo;
    
%    disp(sqrt(sumerr/NTD))
%     save -ascii Wi.dat Wi;
%     save -ascii Wo.dat Wo;
end


% Cross Validate the network.....................................................
CVrmstra = zeros(out,1);
CVres_tra = zeros(NCV,2*out);

CVcorrect = 0;
for sa = NTD+1 : TD
        
        xx = Ntrain(sa,1:inp)';     % Current Sample
        ttCur = Ntrain(sa,inp+1:end)'; % Current Target
        tt = Ytrue(sa, :)';          %coded output
        
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        
        CVrmstra = CVrmstra + (tt-Yo).^2;
        CVres_tra(sa,:) = [tt' Yo'];
        
        [val, class] = max(Yo);
        
        if ttCur == class
            CVcorrect = CVcorrect + 1;
        end
end
disp('cross validation')
disp(sqrt(CVrmstra/NCV))      % mean square error on testing the trained data


% Testing over training data.....................................................
conf_mat_tra = zeros(out, out);

rmstra = zeros(out,1);
res_tra = zeros(TD,2*out);

predict_tra = zeros(TD,2);

%correct_tra = 0;

for sa = 1 : TD
        
        xx = Ntrain(sa,1:inp)';     % Current Sample
        ttCur = Ntrain(sa,inp+1:end)'; % Current Target
        tt = Ytrue(sa, :)';          %coded output
        
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        
        rmstra = rmstra + (tt-Yo).^2;
        res_tra(sa,:) = [tt' Yo'];
        
        [val, class] = max(Yo);
        
        conf_mat_tra(ttCur, class) = conf_mat_tra(ttCur, class) + 1;
        
        predict_tra(sa,:) = [ttCur class];
        
%        if ttCur == class
%            correct_tra = correct_tra + 1;
%        end
end
disp('training validation')
disp(sqrt(rmstra/TD))      % mean square error on testing the trained data
%disp(conf_mat_tra)

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


% Test the network.........................................................
NFeature=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Set 5\Iris.tes');
[NTD,~]=size(NFeature);

NAns=load('D:\SEMESTER_6\IT481_Topics_In_Neural_Networks\Neural Network Dec 2016\Assignment Classification\Results\Group 5\Iris.cla');

% create the output vectors for the true/actual outputs
YAns = zeros(NTD, Nclasses);
for i = 1 : NTD
   YAns(i, :) = -1;
   % can take another loop to traverse column-wise
   YAns(i, NAns(i, 1)) = 1;     %%%%%%%%%%%%%%%CHECK matrix true value how to convert ?????
end

conf_mat_tes = zeros(out, out);

rmstes = zeros(out,1);
res_tes = zeros(NTD,2*out);

predict_tes = zeros(NTD,2);

%correct_ans = 0;

for sa = 1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        ca = NAns(sa);      % Actual Output
        tt = YAns(sa, :)';          %coded output
        
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output

        rmstes = rmstes + (tt-Yo).^2;
        res_tes(sa,:) = [tt' Yo'];
        
        [val, class] = max(Yo);
        
        conf_mat_tes(ca, class) = conf_mat_tes(ca, class) + 1;
        
        predict_tes(sa,:) = [ca class];
        
%        if ca == class
%            correct_ans = correct_ans + 1;
%        end
end
disp('testing')
disp(sqrt(rmstes/NTD))
%disp(conf_mat_tes)

%correct classifications
correct_tes = sum(diag(conf_mat_tes));

%overall accuracy
overall_acc_tes = 100*correct_tes/NTD;


%to get average accuracy

% using histogram to get frequency of labels
[labelCount_tes,classLabel] = hist(NAns, unique(classLabel));

%average accuracy
avg_acc_tes = 100/out * sum(diag(conf_mat_tes)./labelCount_tes');

%geometric-mean accuracy
geo_mean_acc_tes = nthroot(prod(100*diag(conf_mat_tes)./labelCount_tes'),out);

