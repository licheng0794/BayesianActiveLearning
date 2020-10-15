% plot the original 2D function
figure;
[x1, x2] = meshgrid(linspace(-5, 5, num_points));
mesh(x1,x2,reshape(y_star, num_points, num_points));
xlabel('x1');
ylabel('x2');
zlabel('y');
figure;
loglenscale = log(length_scale);
d = length(length_scale);
x = [-3:.1:3];
iter = [1 5 10 15 30];
for i = iter
   RandomUnc = inv(HnlZ{i}.value);
   if i==30
       newi = 5;
   else
       newi = round(i/5+1);
   end
   subplot(3, 5, newi); % for the active learning
   for id = 1:d
       
       y = normpdf(x,results.map_hyperparameters(i).cov(id),results.secondSigma{i}(id,id));
       if id ==1
           plot(x,y, 'b'); hold on
           ylim1=get(gca,'Ylim'); 
           
       else
           plot(x,y, 'r'); hold on
           ylim2=get(gca,'Ylim');
           ymax = max(ylim1, ylim2);
           ylim = ymax;
           plot([loglenscale(1),loglenscale(1)],ylim,'b--'); hold on;
           plot([loglenscale(2),loglenscale(2)],ylim,'r--'); hold on;
           if i==1
                title('n=2')
           else
                title(['n=' num2str(i)]);
           end
       end
   end
   
   subplot(3, 5, newi+5); % for uncertainty sampling
   for id = 1:d
       
       y = normpdf(x,results_unc.map_hyperparameters(i).cov(id),results_unc.secondSigma{i}(id,id));
       if id ==1
           plot(x,y, 'b'); hold on
           ylim1=get(gca,'Ylim'); 
           
       else
           plot(x,y, 'r'); hold on
           ylim2=get(gca,'Ylim');
           ymax = max(ylim1, ylim2);
           ylim = ymax;
           plot([loglenscale(1),loglenscale(1)],ylim,'b--'); hold on;
           plot([loglenscale(2),loglenscale(2)],ylim,'r--'); hold on;
       end
   end
   
   subplot(3, 5, newi+10); % for random sampling
   for id = 1:d
       
       y = normpdf(x,map_hyperparameters_random{i}.cov(id),RandomUnc(id,id));
       if id ==1
           plot(x,y, 'b'); hold on
           ylim1=get(gca,'Ylim'); 
          
       else
           plot(x,y, 'r'); hold on
           ylim2=get(gca,'Ylim'); 
           ymax = max(ylim1, ylim2);
           ylim = ymax;
           plot([loglenscale(1),loglenscale(1)],ylim,'b--'); hold on;
           plot([loglenscale(2),loglenscale(2)],ylim,'r--'); hold on;
       end
   end
end