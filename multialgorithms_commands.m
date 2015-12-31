function multialgorithms_commands(command)

%This function processes events from the multi-algorithm GUI screen

switch(command)
   
case {'MoveLeft','MoveRight'}
   %Move algorithms between the list boxes
   if strcmp(command,'MoveLeft'),
      hFrom = findobj('Tag','lstAllAlgorithms');
      hTo 	= findobj('Tag','lstChosenAlgorithms');
   else
      hTo   = findobj('Tag','lstAllAlgorithms');
      hFrom	= findobj('Tag','lstChosenAlgorithms');
   end
   
   %Find the selected algorithm and remove it from the 'From list'
   val		     = get(hFrom,'Value');     
   algorithms    = get(hFrom,'String'); 
   
   algorithm	  = algorithms(val,:);
   if (isempty(deblank(algorithm)))
      break
   end
   
   newlist 		  = 1:size(algorithms,1);
   newlist(val)  = 0;
   newlist		  = newlist(find(newlist ~=0));
   
   set(hFrom, 'Value', 1);
   if ~isempty(newlist),
      set(hFrom, 'String', algorithms(newlist,:));
   else
      set(hFrom, 'String', '          ');
   end
   
   %Put the new algorithm in the 'To list'
   algorithms    = get(hTo,'String'); 
   L				  = max(size(algorithms,2),size(algorithm,2));
   if ((isempty(deblank(algorithms(1,:)))) & (size(algorithms,1) == 1))
      newalgorithms = algorithm;
   else
      newalgorithms = zeros(size(algorithms,1)+1,L);
      newalgorithms(1:size(algorithms,1),:) = algorithms;
      newalgorithms(size(algorithms,1)+1,:) = algorithm;
   end
   
   set(hTo, 'String', char(newalgorithms))
   
case {'Compare', 'Predict'}
	Npoints = 100;

	hFigure = gcf;
	hm = findobj('Tag', 'Messages'); 
	set(hm,'String','');     
   
   %Do some error checking
   if evalin('base', '~exist(''targets'')')    
      set(hm,'String','No targets on workspace. Please load targets.')   
      break
   end
   
   if evalin('base', '~exist(''features'')')    
      set(hm,'String','No features on workspace. Please load features.')
      break
   end 
   
   features                = evalin('base','features');
   targets                 = evalin('base','targets');
   if (evalin('base', 'exist(''distribution_parameters'')')),
      distribution_parameters = evalin('base', 'distribution_parameters');
   end
   
   %Find the region for the grid
   [region,x,y]  = calculate_region(features, [zeros(1,4) Npoints]);    
      
   %Find which algorithms will be used
   hAlgorithms	= findobj('Tag','lstChosenAlgorithms');
   algorithms  = get(hAlgorithms,'String'); 
   
   if ((isempty(deblank(algorithms(1,:)))) & (size(algorithms,1) == 1))
      set(hm,'String','Please select at least one algorithm.')     
      break   
   end
   Nalgorithms = size(algorithms,1);
   
   All_algorithms = read_algorithms('Classification.txt');
   
   for i = 1:Nalgorithms,
      index = strmatch(deblank(algorithms(i,:)),char(All_algorithms(:).Name),'exact');
      if ~isempty(index),
         Chosen_algorithms(i).Name = deblank(algorithms(i,:));
         if isempty(strmatch('N',All_algorithms(index).Field)),
            Chosen_algorithms(i).Parameter = char(inputdlg(['Enter ' All_algorithms(index).Caption], All_algorithms(index).Name, 1, cellstr(All_algorithms(index).Default)));
         else
            Chosen_algorithms(i).Parameter = '';
         end
      end
   end
   
   if strcmp(command, 'Compare'),
      %Comapre the algorithms      
      error_method_val = get(findobj('Tag', 'popErrorEstimation'),'Value');
      error_method_str = get(findobj('Tag', 'popErrorEstimation'),'String');
      error_method 	  = char(error_method_str(error_method_val));
      
      h = findobj('Tag', 'txtRedraws');   
      redraws = str2num(get(h, 'String'));   
      if isempty(redraws), 
         set(hm,'String','Please select how many redraws are needed.')      
         break
      else     
         if strcmp(error_method, 'Cross-Validation'),
            if (redraws < 2),
               set(hm, 'String', 'Number of redraws must be larger than 1.')
               break
            end
         else
            if (redraws < 1), 
               set(hm,'String','Number of redraws must be larger than 0.')     
               break    
            end
         end   
      end      
      
      h = findobj('Tag', 'txtPrecentage'); 
      percent = str2num(get(h, 'String'));   
      if strcmp(error_method, 'Holdout'),
         if isempty(percent), 
            set(hm,'String','Please select the percentage of training vectors.')     
            break
         else     
            if (floor(percent/100*length(targets)) < 1),     
               set(hm,'String','Number training vectors must be larger than 0.')     
               break    
            end   
         end
      end              
      
      %Now that the data is OK, start working
      set(gcf,'pointer','watch');
      
      %Some variable definitions
      Nclasses		= find_classes(targets); %Number of classes in targets
      test_err 		= zeros(Nalgorithms,redraws);   
      train_err 		= zeros(Nalgorithms,redraws);   
      
      for k = 1: Nalgorithms,
         for i = 1: redraws,  
            set(hm, 'String', [Chosen_algorithms(k).Name ' algorithm: Processing iteration ' num2str(i) ' of ' num2str(redraws) ' iterations...']);
            
            %Make a draw according to the error method chosen
            L = length(targets);
            switch error_method
            case cellstr('Resubstitution')
               test_indices = 1:L;
               train_indices = 1:L;
            case cellstr('Holdout')
               [test_indices, train_indices] = make_a_draw(floor(percent/100*L), L);           
            case cellstr('Cross-Validation')
               chunk = floor(L/redraws);
               test_indices = 1 + (i-1)*chunk : i * chunk;
               train_indices  = [1:(i-1)*chunk, 1+i * chunk:L];
            end
            
            train_features = features(:, train_indices);    
            train_targets  = targets (:, train_indices);    
            test_features  = features(:, test_indices);     
            test_targets   = targets (:, test_indices);     
            
            param = str2num(Chosen_algorithms(k).Parameter);
            if isempty(param),
               param = Chosen_algorithms(k).Parameter;
            end
            D = feval(Chosen_algorithms(k).Name, train_features, train_targets, param, region); 
            
            [classify, err]  = classification_error(D, train_features, train_targets, region);
            train_err(k,i)   = err;     
            [classify, err]  = classification_error(D, test_features, test_targets, region);
            test_err(k,i)    = err;     
            
         end      
      end
      
      hDisp   = findobj('Tag','popErrorDisplay');
      sDisp   = get(hDisp,'String');
      switch char(sDisp(get(hDisp,'Value'))),
      case 'Test error'
         if (redraws > 1),
            err    = mean(test_err');
         else
            err    = test_err;
         end        
      case 'Train error'
         if (redraws > 1),
            err    = mean(train_err');
         else
            err    = train_err';
         end
      otherwise
         if (redraws > 1),
            err = mean(test_err')*length(test_targets)+mean(train_err')*length(train_targets);
            err = err / (length(test_targets)+length(train_targets));
         else
            err = test_err*length(test_targets)+train_err*length(train_targets);
            err = err / (length(test_targets)+length(train_targets));
         end   
      end
      
      hBayes = findobj('Tag','chkBayes');
      if ((get(hBayes, 'Value')) & (exist('distribution_parameters'))),
         if ~isempty(distribution_parameters)
            Dbayes = decision_region(distribution_parameters, region);
            [classify, Bayes_err]  = classification_error(Dbayes, features, targets, region);
            err(length(err)+1) = Bayes_err;
            Nalgorithms = Nalgorithms + 1;
            Chosen_algorithms(Nalgorithms).Name='Bayes err       ';
         end
      end   
      
      %Plot the results
      figure
      bar(err)
      title('Average classification errors')
      for k=1:Nalgorithms,
         str = deblank(Chosen_algorithms(k).Name);
         str(findstr(str,'_')) = ' ';
         h=text(k,err(k)+.02,str);
         set(h,'HorizontalAlignment','Center')
         set(h,'FontSize',12)
         %set(h,'Color',[1 1 1])
      end
      ax = axis;ax(3)=0;ax(4)=max(1,max(err));
      axis(ax)
      
      s = 'Finished!';
      set(hm, 'String', s);   
      set(hFigure,'pointer','arrow');
      assignin('base','final_errors',err)
   else    
      %Predict performance
      a   = zeros(1, Nalgorithms);
      
      for k = 1: Nalgorithms,
         set(hm, 'String', [Chosen_algorithms(k).Name ' algorithm']);
         
         param = str2num(Chosen_algorithms(k).Parameter);
         if isempty(param),
            param = Chosen_algorithms(k).Parameter;
         end
         a(k) = predict_performance(Chosen_algorithms(k).Name, param, features, targets, region); 
      end
      
      %Plot the results
      figure
      bar(a)
      title('Prediction values')
      for k=1:Nalgorithms,
         str = deblank(Chosen_algorithms(k).Name);
         str(findstr(str,'_')) = ' ';
         h=text(k,a(k)+.02,str);
         set(h,'HorizontalAlignment','Center')
         set(h,'FontSize',12)
         %set(h,'Color',[1 1 1])
      end
      assignin('base','final_predictions',a)
      
   end
   
otherwise
   error('Unknown commands')
end
