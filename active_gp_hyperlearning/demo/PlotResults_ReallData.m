clear
x=1:34;
% load activeGP_05.mat
% std_dev_300 = diag(results.secondSigma{500});
% load uncSampling_05.mat
% std_dev_400 = diag(results_unc.secondSigma{500});
% y = [ std_dev_300(x) std_dev_400(x)];  
% figure
% bar(y)
% legend( 'BAL 500 points',  'UNC 500 points')
% 
% xText = {'blue-jet-speed',	'red-jet-speed',	'blue-jet-sensor-range',	'red-jet-sensor-range',...
%     'blue-AEW&C-speed',	'red-AEW&C-speed',	'blue-AEW&C-sensor-range',	'red-AEW&C-sensor-range',...
%     'blue-GBAD-speed',	'red-GBAD-speed',	'blue-GBAD-sensor-range', ...
%     'red-GBAD-sensor_range',	'blue-GBAD-weapon-range',	'red-GBAD-weapon-range',...
%     'blue-AWD-speed',	'red-AWD-speed',	'blue-AWD-sensor-range',	'red-AWD-sensor-range',	...
%     'blue-AWD-weapon_range',	'red-AWD-weapon-range',	'blue-subs-speed',...
%     'red-subs-speed',	'blue-subs-sensor-range',	'red-subs-sensor-range',...
%     'blue-subs-weapons-range',	'red-subs-weapons-range',	'AIM120-PK',	'AIM120-Range',...
%     'AIM9-PK',	'AIM9-Range',	'AGM88-PK',	 'AGM88-Range',	 'AWD-PK',	'GBAD-PK' ...
% };
% xticks(x)
% xticklabels(xText);
% xtickangle(90)
% ylabel('variance')




% % map400 = 1./exp(results.map_hyperparameters(150).cov(x));
% map500 = 1./exp(results.map_hyperparameters(250).cov(x));
% y = [map500];  
% figure
% bar(y, 'r')
% xticks(x)
% xticklabels(xText);
% xtickangle(90)
% ylabel('variable importance')


% x=1:34;
% y = 1./exp(results_unc.map_hyperparameters(500).cov(1:34));
% 
% 
% std_dev_300 = diag(results.secondSigma{250});
% std_dev_500 = diag(results_unc.secondSigma{250});
% y = [ std_dev_300(x) std_dev_500(x)];  
% figure
% bar(y)
% legend( 'BAL 250 points',  'UNC 250 points')
% 


xText = {'blue-jet-speed',	'red-jet-speed',	'blue-jet-sensor-range',	'red-jet-sensor-range',...
    'blue-AEW&C-speed',	'red-AEW&C-speed',	'blue-AEW&C-sensor-range',	'red-AEW&C-sensor-range',...
    'blue-GBAD-speed',	'red-GBAD-speed',	'blue-GBAD-sensor-range', ...
    'red-GBAD-sensor_range',	'blue-GBAD-weapon-range',	'red-GBAD-weapon-range',...
    'blue-AWD-speed',	'red-AWD-speed',	'blue-AWD-sensor-range',	'red-AWD-sensor-range',	...
    'blue-AWD-weapon_range',	'red-AWD-weapon-range',	'blue-subs-speed',...
    'red-subs-speed',	'blue-subs-sensor-range',	'red-subs-sensor-range',...
    'blue-subs-weapons-range',	'red-subs-weapons-range',	'AIM120-PK',	'AIM120-Range',...
    'AIM9-PK',	'AIM9-Range',	'AGM88-PK',	 'AGM88-Range',	 'AWD-PK',	'GBAD-PK' ...
};
% xticks(x)
% xticklabels(xText);
% xtickangle(90)
% ylabel('variance of parameter importance')


% legend('300', '500')


% x = linspace(-30,30,1000);
% y = normpdf(x,0,5);
% plot(x,y)

load finalResultsGP.mat
map500 = 1./exp(results.map_hyperparameters(1).cov(x));
map500 = map500./max(map500);
y = [map500]; 

figure
subplot(3,1,1)
bar(y, 'b')
% xticks(x)
% xticklabels(xText);
% xtickangle(90)
xticklabels([])
ylabel('VI')
legend('GP 1855 points')




load activeGP_05.mat
map500 = 1./exp(results.map_hyperparameters(500).cov(x));
map500 = map500./max(map500);
y = [map500];
subplot(3,1,2)
bar(y, 'r')
% xticks(x)
% xticklabels(xText);
% xtickangle(90)
xticklabels([])
ylabel('VI')
legend('BAL 500 points')

load uncSampling_05.mat
map500 = 1./exp(results_unc.map_hyperparameters(500).cov(x));
map500 = map500./max(map500);
y = [map500]; 

subplot(3,1,3)
bar(y, 'g')
xticks(x)
xticklabels(xText);
xtickangle(90)
ylabel('VI')
legend('UNC 500 points')



%% compare KL distance between active GP and uncertainty, random sampling and final GP
load finalResultsGP.mat
meanGP = results.map_hyperparameters(1).cov(x);
varGP = diag(diag(results.secondSigma{1}(x,x)));

load activeGP_05.mat
load uncSampling_05.mat
load RandomResults.mat

iter = [50:50:500];
nn = length(iter);
KL_GP_BAL = zeros(1,nn);
KL_GP_Unc = zeros(1,nn);
KL_GP_Rnd = zeros(1,nn);
ni = 1;
for i = iter
meanBAL = results.map_hyperparameters(i).cov(x);
varBAL = diag(diag(results.secondSigma{i}(x,x)));


meanUnc = results_unc.map_hyperparameters(i).cov(x);
varUnc = diag(diag(results_unc.secondSigma{i}(x,x)));

meanRnd = results_rnd.map_hyperparameters(i).cov(x);
varRnd = diag(diag(results_rnd.secondSigma{i}(x,x)));

KL_GP_BAL(ni) = 0.5.*(log(det(varBAL)/det(varGP)) - 34 + trace(inv(varBAL)*varGP) ...
           + (meanBAL-meanGP)'*inv(varBAL)*(meanBAL-meanGP));

KL_GP_Unc(ni) = 0.5.*(log(det(varUnc)/det(varGP)) - 34 + trace(inv(varUnc)*varGP) ...
           + (meanUnc-meanGP)'*inv(varUnc)*(meanUnc-meanGP));

KL_GP_Rnd(ni) = 0.5.*(log(det(varRnd)/det(varGP)) - 34 + trace(inv(varRnd)*varGP) ...
           + (meanRnd-meanGP)'*inv(varRnd)*(meanRnd-meanGP));       
ni = ni + 1;       
end

figure;
plot(iter, KL_GP_BAL, '-rs');
hold on;
plot(iter, KL_GP_Unc, '-go');
hold on;
plot(iter, KL_GP_Rnd, '-m*');
hold on;

set(gca, 'fontsize', 12);

xlabel('number of iterations');
ylabel('KL divergence')
legend('BAL', 'UNC', 'Random');