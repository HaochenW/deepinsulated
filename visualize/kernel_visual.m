% for i = 1:4
%     for j = 1:4
%         if i < j
%             %% load the experiment data to plot the datapoints (1252 points)
%             A = [i j];
%             formatSpec = "original_mat_Kernel %d and Kernel %d.mat";
%             str = compose(formatSpec,A);
%             [x,y,z] = load_data(str);
%             z = log2(z);
%             figure;
%             plot3(x,y,z,'o', 'Color','#ffffb3','MarkerSize',2)
%             scatter3(x,y,z,10,z, 'filled')
% 
%             hold on;
%             
%             formatSpec = "mat_Kernel %d and Kernel %d.mat";
%             str = compose(formatSpec,A);
%             S = load(str);
%             X = S.name1;
%             Y = S.name2;
%             Z = S.name3;
% %             mesh(X,Y,Z);
%             surf(X, Y, Z);
%             colormap(jet);
%             alpha 0.5
%             colorbar();
%             formatSpec = "Kernel_%d_and_Kernel_%d.fig";
%             A = [i-1 j-1];
%             str = compose(formatSpec,A);
%             savefig(str)
%             formatSpec = "Kernel_%d_and_Kernel_%d.png";
%             str = compose(formatSpec,A);
%             saveas(gcf,str)
%         end
%     end
% end
% 
% function [x,y,z] = load_data(data_dir)
%      S = load(data_dir);
%      x = double(S.name1)';
%      y = double(S.name2)';
%      z = double(S.name3)';
% end
for i = 1:4
    for j = 1:4
        if i < j && i == 1 && j == 3
            %% Calculate the mesh grid and draw the surf plot
            formatSpec = "original_mat_Kernel %d and Kernel %d.mat";
%             formatSpec = "mat_Kernel %d and Kernel %d_experiment.mat";
            A = [i j];
            str = compose(formatSpec,A);
            [x,y,z] = load_data(str);
%             z = log2(z);
            grid on
            outside_x = (max(x) - min(x))/4;
            outside_y = (max(y) - min(y))/4;
            grid_size = 50;
            middle_x = (min(x) + max(x))/100;
            middle_y = (min(y) + max(y))/100;
            middle_x = 0;
            middle_y = 0;
            xv = linspace(min(x) - outside_x + middle_x, max(x) + outside_x, grid_size);
            yv = linspace(min(y) - outside_y + middle_y, max(y) + outside_y, grid_size);
            [X,Y] = meshgrid(xv, yv);
            figure(i*10+j)
%             Z = griddata(x,y,z,X,Y,'linear');
            f = scatteredInterpolant(x,y,z, 'linear', 'none');
            f.Method = 'natural';
            Z = f(X,Y);
            Z(find(isnan(Z)==1)) = 0;
            
            % Draw the surf plot
            colormap;
            cmaps{1} = colormap();
            h(1) = surf(X, Y, Z);
%             contourf(X,Y,Z)
%             mesh(X, Y, Z);
            grid on
            alpha 0.5
            colorbar;
            hold on
             %% load the second experiment data to plot the datapoints (112 points)
            formatSpec = "original_mat_experiment_Kernel %d and Kernel %d.mat";
            str = compose(formatSpec,A);
            [x,y,z] = load_data(str);
            x = x(61:70);
            y = y(61:70);
            z = z(61:70);
            z = z * 10000;
%             plot3(x,y,z,'.', 'Color','#ff3300','MarkerSize',35)
%             scatter3(x,y,z,10,z, 'filled','MarkerEdgeColor','k')
%             colorbar
%             hold on;
            %% load the experiment data to plot the datapoints (1252 points)
            formatSpec = "original_mat_Kernel %d and Kernel %d.mat";
            str = compose(formatSpec,A);
            [x,y,z] = load_data(str);
            z = log2(z);
            plot3(x,y,z,'o', 'Color','#ffffb3','MarkerSize',2)
            scatter3(x,y,z,10,z, 'filled')
            colorbar
            %% load the test experiment data to plot the link sequence data 
            test_linkseq_file = 'link_seq.mat';
            feature_seq = load(test_linkseq_file).name1;
        
            draw_start = 1; %8,16   3,15
            draw_end = 1;
            test_feature_seq = feature_seq(1,draw_start);
            test_feature1_seq = double(max(test_feature_seq{1,1}(:,:,i),[],2));
            test_feature2_seq = double(max(test_feature_seq{1,1}(:,:,j),[],2));
            test_feature3_seq = f(test_feature1_seq,test_feature2_seq)  * 10;
            test_feature3_seq(find(isnan(test_feature3_seq)==1)) = 0;
            
            
            % 去掉一些样本点
%             index = [2,5,9,13];
%             index = [3,6,8,14];
%             index =  [6,8,12,15];%保留的样本点
%             test_feature1_seq =  test_feature1_seq(index);
%             test_feature2_seq =  test_feature2_seq(index);
%             test_feature3_seq =  test_feature3_seq(index);
            
            
            
%             g1 = plot3(test_feature1_seq,test_feature2_seq,test_feature3_seq,'.', 'Color','black','MarkerSize',20);
            g2 = plot3(test_feature1_seq,test_feature2_seq,test_feature3_seq,'-', 'Color','black',  'LineWidth',2.0);
%             hold on;
            %% load the test experiment data to plot the datapoints (60 points)
            test_experimentfile = 'start_end.mat';
            start_point = double(load(test_experimentfile).name1);
            end_point = double(load(test_experimentfile).name2);
%             feature1_start = test_feature1_seq(1);
%             feature2_start = test_feature2_seq(1);
%             feature3_start = test_feature3_seq(1);
            feature1_start = max(start_point(draw_start:draw_end,:,i),[],2);
            feature2_start = max(start_point(draw_start:draw_end,:,j),[],2);
            feature3_start = f(feature1_start,feature2_start) * 10;
            feature3_start(find(isnan(feature3_start)==1)) = 0;
            feature1_end = max(end_point(draw_start:draw_end,:,i),[],2);
            feature2_end = max(end_point(draw_start:draw_end,:,j),[],2);
            feature3_end = f(feature1_end,feature2_end) * 10;
            feature3_end(find(isnan(feature3_end)==1)) = 0;
            
            g3 = plot3(feature1_start,feature2_start,feature3_start,'.', 'Color','#99ffff','MarkerSize',20);
            g4 = plot3(feature1_end,feature2_end,feature3_end,'.', 'Color','#e41a1c','MarkerSize',20);
%             g4 = scatter3(feature1_end,feature2_end,feature3_end,50,feature3_end, 'filled');
%             for k = 1: length(feature1_start)
%                 g5 = plot3([feature1_start(k),feature1_end(k)], [feature2_start(k), feature2_end(k)], [feature3_start(k), feature3_end(k)], 'Color','black', 'LineWidth',1)
%             end
% %             legend([g3, g4, g1, g2], {'start','end', 'optimized points','optimized ways'}, 'Location','best',...
% %                                   'NumColumns',1,'FontSize',12,'TextColor','black');
%             legend([g3, g4, g5], {'start','end', 'matching lines'}, 'Location','best',...
%                                   'NumColumns',1,'FontSize',12,'TextColor','black');
           

           %% Random density
            str = "random_mat_Kernel 1 and Kernel 3.mat";
            [x,y] = load_data_random(str);
            % Estimate a continuous pdf from the discrete data
            [pdfx xi]= ksdensity(x);
            [pdfy yi]= ksdensity(y);
            % Create 2-d grid of coordinates and function values, suitable for 3-d plotting
            [xxi,yyi]     = meshgrid(xi,yi);
            [pdfxx,pdfyy] = meshgrid(pdfx,pdfy);
            % Calculate combined pdf, under assumption of independence
            pdfxy = pdfxx.*pdfyy; 
            % Plot the results
            [M,c] = contour(xi,yi,pdfxy);
            c.LineWidth = 3;
%             c.color = 'red'
            % axis off;
%             set(gca,'XLim',[min(xi) max(xi)])
%             set(gca,'YLim',[min(yi) max(yi)])


            
            %% Save the picture
            formatSpec = "Kernel_%d_and_Kernel_%d.fig";
            A = [i-1 j-1];
            str = compose(formatSpec,A);
            savefig(str)
            formatSpec = "Kernel_%d_and_Kernel_%d.png";
            str = compose(formatSpec,A);
            set(gcf,'PaperType','a3');
            saveas(gcf,str)
        end
    end
end

function [x,y,z] = load_data(data_dir)
     S = load(data_dir);
     x = double(S.name1)';
     y = double(S.name2)';
     z = double(S.name3)';
end



function [x,y] = load_data_random(data_dir)
     S = load(data_dir);
     x = double(S.name1)';
     y = double(S.name2)';
end

