function classifier_commands(command)

%This function processes events from the main (single-algorithm) GUI screen

persistent Preprocessing_methods;
persistent Algorithms;
if (isempty(Preprocessing_methods) | isempty(Algorithms)),
    Algorithms = read_algorithms('Classification.txt');
    Preprocessing_methods = read_algorithms('Preprocessing.txt');
end

switch(command)
    
case 'Init'
    %Init of the classifier GUI
    h				= findobj('Tag', 'Preprocessing');
    set(h,'String',strvcat(Preprocessing_methods(:).Name));
    chosen = strmatch('None',char(Preprocessing_methods(:).Name));
    set(h,'Value',chosen);
    
    h				= findobj('Tag', 'Algorithm');
    set(h,'String',strvcat(Algorithms(:).Name));
    chosen = strmatch('LS',char(Algorithms(:).Name));
    set(h,'Value',chosen);
    
case 'Changed Preprocessing'
    h				= findobj(gcbf, 'Tag', 'Preprocessing');
    chosen       	= get(h, 'Value');
    
    hLabel	  		= findobj(gcbf, 'Tag', 'lblPreprocessingParameters');
    hBox		  		= findobj(gcbf, 'Tag', 'txtPreprocessingParameters');
    set(hBox,'Visible','on');
    
    if ~isempty(chosen),
        set(hLabel,'String',Preprocessing_methods(chosen).Caption);
        set(hBox,'String',Preprocessing_methods(chosen).Default);
        if strcmp(char(Preprocessing_methods(chosen).Field),'N')
            set(hLabel,'String','');
            set(hBox,'Visible','off');    
        end
    else
        set(hLabel,'String','');
        set(hBox,'Visible','off');    
    end
    
case 'Changed Algorithm'
    h					= findobj(gcbf, 'Tag', 'Algorithm');
    chosen             	= get(h, 'Value');
    
    hLabel	  		= findobj(gcbf, 'Tag', 'lblAlgorithmParameters');
    hBox	 		= findobj(gcbf, 'Tag', 'txtAlgorithmParameters');
    hLongBox		= findobj(gcbf, 'Tag', 'txtAlgorithmParametersLong');
    set(hBox,'Visible','on');
    
    if ~isempty(chosen),
        set(hLabel,'String',Algorithms(chosen).Caption);
        switch Algorithms(chosen).Field
        case 'S'
            set(hBox,'String',Algorithms(chosen).Default);
            set(hBox,'Visible','on');
            set(hLongBox,'Visible','off');
        case 'L'
            set(hLongBox,'String',Algorithms(chosen).Default);
            set(hBox,'Visible','off');
            set(hLongBox,'Visible','on');
        case 'N'
            set(hLabel,'String','');
            set(hBox,'Visible','off');    
            set(hLongBox,'Visible','off');
        end
    else
        set(hLabel,'String','');
        set(hBox,'Visible','off');    
        set(hLongBox,'Visible','off');
    end
    
case 'Start'
    
    %Start the classification process
    Npoints = 100; %Number of points on each axis of the decision region

    %Read data from the workspace
    hm = findobj('Tag', 'Messages'); 
    hParent = get(hm,'Parent'); %Get calling window tag
    set(hm,'String',''); 
    
    h =  findobj('Tag', 'TestSetError');
    set(h, 'String', '');
    h =  findobj('Tag', 'TrainSetError');
    set(h, 'String', '');
    
    %Do some error checking
    if evalin('base', '~exist(''targets'')')    
        set(hm,'String','No targets on workspace. Please load targets.')   
        return
    end
    
    if evalin('base', '~exist(''features'')')    
        set(hm,'String','No features on workspace. Please load features.')
        return
    end 

    features                = evalin('base','features');
    targets                 = evalin('base','targets');    
    if (evalin('base', 'exist(''distribution_parameters'')')),
	    distribution_parameters = evalin('base', 'distribution_parameters');
    end
    
    error_method_val = get(findobj('Tag', 'popErrorEstimationMethod'),'Value');
    error_method_str = get(findobj('Tag', 'popErrorEstimationMethod'),'String');
    error_method 	 = char(error_method_str(error_method_val(1)));
        
    h = findobj('Tag', 'Redraws');   
    redraws = str2num(get(h, 'String'));   
    if isempty(redraws), 
        set(hm,'String','Please select how many redraws are needed.')      
        return
    else     
        if (redraws < 1), 
            set(hm,'String','Number of redraws must be larger than 0.')     
            return    
        else
            if (strcmp(error_method, 'Cross-Validation') & (redraws == 1)),
                set(hm,'String','Number of redraws must be larger than 1.')     
                return    
            end        
        end   
    end      
    
    h = findobj('Tag', 'PercentTraining'); 
    percent = str2num(get(h, 'String'));   
    if strcmp(error_method, 'Holdout'),
        if isempty(percent), 
            set(hm,'String','Please select the percentage of training vectors.')     
            return
        else     
            if (floor(percent/100*length(targets)) < 1),     
                set(hm,'String','Number of training vectors must be larger than 0.')     
                return    
            end   
            if (percent >= 100),     
                set(hm,'String','Number of training vectors must be smaller than 100%.')     
                return    
            end   
            if (floor((1-percent/100)*length(targets)) < 1),     
                set(hm,'String','Number of test vectors must be larger than 0.')     
                return    
            end   
        end
    end      
    
    %Find the region for the grid
    [region,x,y]  = calculate_region(features, [zeros(1,4) Npoints]);    
    
    h		      = findobj('Tag', 'Algorithm'); 
    val		      = get(h,'Value');     
    algorithm     = get(h,'String'); 
    algorithm 	  = deblank(char(algorithm(val,:)));
    h		      = findobj('Tag', 'Preprocessing'); 
    val		      = get(h,'Value');     
    preprocessing = get(h,'String'); 
    preprocessing = deblank(char(preprocessing(val,:)));
    
    %Get algorithm paramters. If they don't contain a string, turn them into a number
    if strcmp(get(findobj('Tag', 'txtAlgorithmParameters'),'Visible'),'on'),
        AlgorithmParameters = char(get(findobj('Tag', 'txtAlgorithmParameters'),'String'));
    else
        AlgorithmParameters = char(get(findobj('Tag', 'txtAlgorithmParametersLong'),'String'));
    end
    if (~isempty(str2num(AlgorithmParameters))),
        AlgorithmParameters = str2num(AlgorithmParameters);
    end
    
    if strcmp(get(findobj('Tag', 'txtPreprocessingParameters'),'Visible'),'on'),
        PreprocessingParameters = char(get(findobj('Tag', 'txtPreprocessingParameters'),'String'));
    else
        PreprocessingParameters = char(get(findobj('Tag', 'txtPreprocessingParameters'),'String'));
    end
    if (~isempty(str2num(PreprocessingParameters))),
        PreprocessingParameters = str2num(PreprocessingParameters);
    end
    
    plot_on = strcmp(get(findobj(gcbf,'Label','&Show center of partitions during training'),'Checked'),'on');
    
    SepratePreprocessing = strcmp(get(findobj(hParent, 'Tag', '&Options&SeparatePreprocessing'),'Checked'),'on');
    
    %Now that the data is OK, start working
    set(gcf,'pointer','watch');
    hold off
    plot_scatter(features, targets, hParent)
    axis(region(1:4))
    hold on
    drawnow

    %Call the main classification function
    [D, test_err, train_err, train_features, train_targets, reduced_features, reduced_targets] = ...
        start_classify(features, targets, error_method, redraws, percent, preprocessing, PreprocessingParameters, ...
                       algorithm, AlgorithmParameters, region, hm, SepratePreprocessing, plot_on);

    %Put results on the workspace as well
    assignin('base', 'D', D);
    assignin('base', 'test_err', test_err);
    assignin('base', 'train_err', train_err);
    assignin('base', 'region', region);

    if (~isempty(reduced_features))
        assignin('base', 'reduced_features', reduced_features);
        assignin('base', 'reduced_targets', reduced_targets);
    else
        evalin('base', 'clear reduced_features reduced_targets');
    end        

    %Display error
    Nclasses = length(unique(targets));
    h =  findobj('Tag', 'TestSetError');
    s = 'Test set errors: ';
    for j = 1:Nclasses,
        s = [s 'Class ' num2str(j) ': ' num2str(mean(test_err(j,:)),2) '. '];
    end
    s = [s 'Total: ' num2str(mean(test_err(Nclasses+1,:)),2)];
    set(h, 'String', s);
    
    h =  findobj('Tag', 'TrainSetError');
    
    h =  findobj('Tag', 'TrainSetError');
    s = 'Train set errors: ';
    for j = 1:Nclasses,
        s = [s 'Class ' num2str(j) ': ' num2str(mean(train_err(j,:)),2) '. '];
    end
    s = [s 'Total: ' num2str(mean(train_err(Nclasses+1,:)),2)];
    set(h, 'String', s);
    
    %Show decision region
    if strcmp(get(findobj(gcbf,'Label','Shade &Decision Regions'),'Checked'),'on'),
        [s,h]=contourf(x,y,D,1);colormap([198 198 255; 240 255 240]/255);
    else
        [s,h]=contour(x,y,D,1);
    end
    set(h,'LineWidth',2,'EdgeColor','k');
    set(h,'UserData','DecisionRegion');
    
    %Show Bayes decision region and error (if possible)
    hBayes = findobj('Tag','chkBayes');
    if ((get(hBayes, 'Value')) & (exist('distribution_parameters'))),
        Dbayes = decision_region(distribution_parameters, region);
        [s,h]	 = contour(x,y,Dbayes,1);set(h,'LineWidth',1,'EdgeColor','r');
        [classify, Bayes_err]  = classification_error(Dbayes, features, targets, region);
        h =  findobj('Tag', 'BayesError');
        s = 'Bayes errors: ';
        for j = 1:Nclasses,
            s = [s 'Class ' num2str(j) ': ' num2str(1 - classify(j,j),2) '. '];
        end
        s = [s 'Total: ' num2str(mean(1 - diag(classify)),2)];
        set(h, 'String', s);
        assignin('base', 'Dbayes', Dbayes);    
    end   
    
    %Draw grid if neccessary
    h = findobj('Label', '&Grid');
    if strcmp('on',get(h, 'Checked')),
        grid on
        set(gca,'layer','top')
    else
        grid off
    end
    
    %Replot training points if necessary
    if strcmp(get(findobj(gcbf,'Label','Show &Training points'),'Checked'),'on'),
        plot_scatter(train_features, train_targets, hParent, 2)
    end
    
    hold off
    
    %That's all folks!
    s = 'Finished!';
    if (redraws > 1),
        s = [s ' (Note that only the last decision region is shown)'];
    end
    
    set(hm, 'String', s);   
    set(gcf,'pointer','arrow');
    
case 'Compare'
    %Start the algorithm comparison screen
    h   = multialgorithms;
    h1  = findobj(h, 'Tag', 'lstAllAlgorithms');
    set(h1, 'String', strvcat(Algorithms(:).Name));
    
case 'FileNameInput'
    hold off
    clear distribution_parameters features targets
    h =  findobj('Tag', 'BayesError');
    set(h, 'String', '');
    h =  findobj('Tag', 'TestSetError');
    set(h, 'String', '');
    h =  findobj('Tag', 'TrainSetError');
    set(h, 'String', '');
    evalin('base','region = [0 0 0 0 500];')
    evalin('base','[features, targets, distribution_parameters] = load_file(get(gcbo, ''String''), region);')
    if (evalin('base', 'isempty(distribution_parameters)')),
        evalin('base', 'clear distribution_parameters');
    end
    
    if (evalin('base', '~isempty(features)'))
        evalin('base','region = calculate_region(features, [zeros(1,4) 100]);')
    end
    
case 'SearchForFile'
    evalin('base','region = [0 0 0 0 500];')
    evalin('base','[filename, pathname] = uigetfile(''*.mat'', ''Open file ...'');')
    if (evalin('base','filename ~= 0')),
        clear distribution_parameters features targets
        h =  findobj('Tag', 'BayesError');
        set(h, 'String', '');
        h =  findobj('Tag', 'TestSetError');
        set(h, 'String', '');
        h =  findobj('Tag', 'TrainSetError');
        set(h, 'String', '');
        evalin('base','set(findobj(''Tag'', ''FileNameInput''), ''String'', [pathname filename]);')
        evalin('base','[features, targets, distribution_parameters] = load_file([pathname, filename], region);')
        evalin('base','clear pathname filename');
        if (evalin('base', 'isempty(distribution_parameters)')),
            evalin('base', 'clear distribution_parameters');
        end
        if (evalin('base', '~isempty(features)'))
            evalin('base','region = calculate_region(features, [zeros(1,4) 100]);')
            classifier_commands('ClearBounds');
        else
            %Clean the axis
            cla
        end
    end
    
case 'EnterManualDist'
    disp('Clearing current data from workspace');
    evalin('base', 'clear')
    cla
    
    ttl	= 'Manual definition of distribution';
    n0	= str2num(char(inputdlg('Enter the number of Gaussians in class 0:',ttl,1,cellstr(num2str(1)))));
    n1	= str2num(char(inputdlg('Enter the number of Gaussians in class 1:',ttl,1,cellstr(num2str(1)))));
    
    distribution_parameters.p0	= str2num(char(inputdlg('Enter the probability of class 0:',ttl,1,cellstr(num2str(0.5)))));
    
    distribution_parameters.m0	= zeros(n0,2);
    distribution_parameters.m1 	= zeros(n1,2);
    distribution_parameters.s0  	= zeros(n0,2,2);
    distribution_parameters.s1  	= zeros(n1,2,2);
    for i = 1:n0, 
        distribution_parameters.s0(i,:,:) = eye(2); 
    end
    for i = 1:n1, 
        distribution_parameters.s1(i,:,:) = eye(2); 
    end
    distribution_parameters.w0 	= ones(n0,1)/n0;
    distribution_parameters.w1 	= ones(n1,1)/n1;
    save synthetic distribution_parameters
    uiwait(enter_distributions);
    evalin('base','region = [0 0 0 0 500];')
    evalin('base','distribution_parameters = load_file(''synthetic'', region);')
        
case 'ManualGraphData'
    %Graphically enter a data set
    evalin('base','if ~(exist(''region'')), region = [-1 1 -1 1 100];end');
    evalin('base','if (sum(abs(region(1:4))) == 0), region = [-1 1 -1 1 100];end');
    evalin('base','[new_features, new_targets, new_params, region] = click_points(region);');   
    new_features = evalin('base','new_features');
    new_targets  = evalin('base','new_targets');
    new_params   = evalin('base','new_params');

    s = questdlg('Do you want to permute the data (Recommended)?','Graphic input','Yes','No','Yes');
    if strcmp(s,'Yes'),
        i               = randperm(length(new_targets));
        new_targets     = new_targets(i);
        new_features    = new_features(:,i);
    end
        
    if (evalin('base','exist(''features'')')),
        features = evalin('base','features');
        targets  = evalin('base','targets');
    else
        features = [];
        targets  = [];
    end
    if (evalin('base','exist(''distribution_parameters'')')),
        distribution_parameters = evalin('base','distribution_parameters');
    else
        distribution_parameters = [];
    end
    
    if isempty(distribution_parameters),
        %If now features exist on the workspace, this is a new distribution, so take it
        %(Otherwise, we may be saving only pars of the distribution)
        if isempty(features)
            distribution_parameters = new_params;
        end
    else
        distribution_parameters.m0=[distribution_parameters.m0; new_params.m0];
        distribution_parameters.m1=[distribution_parameters.m1; new_params.m1];
        distribution_parameters.s0=[distribution_parameters.s0; new_params.s0];
        distribution_parameters.s1=[distribution_parameters.s1; new_params.s1];
        distribution_parameters.w0=[distribution_parameters.w0*sum(~targets); new_params.w0*sum(~new_targets)];
        distribution_parameters.w1=[distribution_parameters.w1*sum( targets); new_params.w1*sum( new_targets)];
        distribution_parameters.w0=distribution_parameters.w0/sum(distribution_parameters.w0);
        distribution_parameters.w1=distribution_parameters.w1/sum(distribution_parameters.w1);
        distribution_parameters.p0=length(distribution_parameters.w0)/(length(distribution_parameters.w0)+length(distribution_parameters.w1));
    end
    
    features = [features, new_features];
    targets  = [targets, new_targets];
    
    assignin('base', 'features', features);
    assignin('base', 'targets', targets);
    assignin('base', 'distribution_parameters', distribution_parameters);
    
    evalin('base','region = calculate_region(features, [zeros(1,4) 100]);')
    
    classifier_commands('ClearBounds');
    
case 'Save'
    evalin('base','[filename, pathname] = uiputfile(''*.mat'', ''Save file ...'');')
    if (evalin('base','filename ~= 0')),
        if (evalin('base', 'exist(''features'')')),
            if (evalin('base', 'exist(''distribution_parameters'')')),
                evalin('base','save([pathname filename], ''features'', ''targets'', ''distribution_parameters'');')
            else
                evalin('base','save([pathname filename], ''features'', ''targets'');')
            end
        else
            if (evalin('base', 'exist(''distribution_parameters'')')),
                evalin('base','save([pathname filename], ''distribution_parameters'');')
            else
                error('No data to save!')
            end
        end         
    end
    
case 'GenerateDistData'
    if (evalin('base','~exist(''distribution_parameters'')')),
        error('The parameters needed to generate the data are missing on the workspace!')
    end
    evalin('base','region = [0 0 0 0 100];')
    evalin('base','[features, targets] = generate_data_set  (distribution_parameters, region);')
    evalin('base','region = calculate_region(features, [zeros(1,4) 100]);')
    
    classifier_commands('ClearBounds');

case 'Params'
    %Print to mean, cov, main directions to workspace 
    if (evalin('base','~isempty(whos(''features''))'))
        
        %There are features on workspace
        features    = evalin('base','features');
        targets     = evalin('base','targets');
        train_one   = find(targets == 1);
        train_zero  = find(targets == 0);
        
        %Estimate mean and covariance for class 0
        m0 = mean(features(:,train_zero)'');
        c0 = cov(features(:,train_zero)'',1);
        p0 = length(train_zero)/length(targets);
        
        %Estimate mean and covariance for class 1
        m1 = mean(features(:,train_one)'');
        c1 = cov(features(:,train_one)'',1);
        
        Chernoff_ = Chernoff(mean(features(:,train_zero)'), cov(features(:,train_zero)',1), mean(features(:,train_one)'), cov(features(:,train_one)',1),length(train_one)/length(targets));
        Bhattacharyya_ = Bhattacharyya(mean(features(:,train_zero)'), cov(features(:,train_zero)',1), mean(features(:,train_one)'), cov(features(:,train_one)',1),length(train_one)/length(targets));
        Discriminability_ = Discriminability(mean(features(:,train_zero)'), cov(features(:,train_zero)',1), mean(features(:,train_one)'), cov(features(:,train_one)',1),length(train_one)/length(targets));
        
        h0 = GaussianParameters;
        set(findobj(h0,'Tag','txtMean00'),'String',num2str(m0(1),3))
        set(findobj(h0,'Tag','txtMean01'),'String',num2str(m0(2),3))
        set(findobj(h0,'Tag','txtMean10'),'String',num2str(m1(1),3))
        set(findobj(h0,'Tag','txtMean11'),'String',num2str(m1(2),3))
        
        set(findobj(h0,'Tag','txtCov000'),'String',num2str(c0(1,1),3))
        set(findobj(h0,'Tag','txtCov001'),'String',num2str(c0(1,2),3))
        set(findobj(h0,'Tag','txtCov010'),'String',num2str(c0(2,1),3))
        set(findobj(h0,'Tag','txtCov011'),'String',num2str(c0(2,2),3))
        set(findobj(h0,'Tag','txtCov100'),'String',num2str(c1(1,1),3))
        set(findobj(h0,'Tag','txtCov101'),'String',num2str(c1(1,2),3))
        set(findobj(h0,'Tag','txtCov110'),'String',num2str(c1(2,1),3))
        set(findobj(h0,'Tag','txtCov111'),'String',num2str(c1(2,2),3))
        
        set(findobj(h0,'Tag','txtBound1'),'String',['The Bhattacharyya bound value is: ' num2str(Bhattacharyya_,3)])
        set(findobj(h0,'Tag','txtBound2'),'String',['The Chernoff bound value is: ' num2str(Chernoff_,3)])
        set(findobj(h0,'Tag','txtDiscriminability'),'String',['The discriminability value is: ' num2str(Discriminability_,3)])
        
        waitfor(h0)
    end
    
case 'MenuZoom'
    umtoggle(findobj(gcbf,'Label','&Zoom'));
    h = findobj(gcbf, 'Label', '&Zoom');
    v = strcmp('on',get(h, 'Checked'));
    if v,
        zoom on
    else                      
        zoom off
    end
    
case 'MenuGrid'
    umtoggle(findobj(gcbf,'Label','&Grid'));
    h = findobj(gcbf, 'Label', '&Grid');
    v = strcmp('on',get(h, 'Checked'));
    if v,
       grid on
       set(gca,'layer','top')
    else
       grid off
    end
    
case 'Copy'
    map = colormap;
    h   = figure;
    copyobj(findobj(gcbf, 'Type','axes'),h);
    set(gca,'Position',[0.1300    0.1100    0.7750    0.8150]);
    set(h, 'colormap', map);
    print -dmeta
    
case 'Print'
    map = colormap;
    h   = figure;
    copyobj(findobj(gcbf, 'Type','axes'),h);
    set(gca,'Position',[0.1300    0.1100    0.7750    0.8150]);
    set(h, 'colormap', map);
    printpreview
    
case 'Exit'
    h = findobj(gcbf, 'Tag', 'Messages');
    set(h,'String','');
    h =  findobj('Tag', 'TestSetError');
    set(h, 'String', '');
    h =  findobj('Tag', 'TrainSetError');
    set(h, 'String', '');
    h =  findobj('Tag', 'FileNameInput');
    set(h, 'String', '');
    h =  findobj('Tag', 'PercentTraining');
    set(h, 'String', '');
    h =  findobj('Tag', 'Redraws');
    set(h, 'String', '');
    h =  findobj('Tag', 'HowMany');
    set(h, 'String', '');
    set(gcf, 'pointer', 'arrow');
    
    close
    
case 'SeparatePreprocessing'
    umtoggle(findobj(gcbf,'Label','&Perform preprocessing separately for each class'));
    
case 'ShowPartitions'
    umtoggle(findobj(gcbf,'Label','&Show center of partitions during training'));
    
case 'ShowTrainingPoints'
    umtoggle(findobj(gcbf,'Label','Show &Training points'));
    
case 'ShadeDecisionRegion'
    umtoggle(findobj(gcbf,'Label','Shade &Decision Regions'));
    
case 'ClearBounds'
    %Clear all bounds
    evalin('base','hold off')
    cla
    if (evalin('base', 'exist(''features'') & exist(''targets'')'))
        evalin('base','plot_scatter(features, targets, gcbf)')
    end
    if (evalin('base', 'exist(''region'')'))
        evalin('base', 'axis(region(1:4))')
    end
    evalin('base', 'drawnow');
    h =  findobj('Tag', 'TestSetError');
    set(h, 'String', '');
    h =  findobj('Tag', 'TrainSetError');
    set(h, 'String', '');
    h = findobj(gcbf, 'Label', '&Grid');
    v = strcmp('on',get(h, 'Checked'));
    if v,
        grid on
        set(gca,'layer','top')
    else
        grid off
    end
    h = findobj('Tag', 'Messages');
    set(h,'String','');
    
case 'ClearWorkspace'
    if strcmp(questdlg('Clear current data on the workspace?', 'Classification toolbox', 'Yes', 'No', 'Yes'), 'Yes')
        %Clear all 
        h =  findobj('Tag', 'TestSetError');
        set(h, 'String', '');
        h =  findobj('Tag', 'TrainSetError');
        set(h, 'String', '');
        h =  findobj('Tag', 'BayesError');
        set(h, 'String', '');
        h =  findobj('Tag', 'FileNameInput');
        set(h, 'String', '');
        hold off
        plot(0,0); grid on;
        h = findobj('Tag', 'Messages');
        set(h,'String','');
        evalin('base','clear all')
    end
    
case 'MixFeatures'
    %Mix (permute) the features
    if (evalin('base','exist(''features'')') & evalin('base','exist(''targets'')')),
        evalin('base','indexes  = randperm(length(targets));')
        evalin('base','features = features(:,indexes);')
        evalin('base','targets  = targets(indexes);')
        h = findobj('Tag', 'Messages');
        set(h,'String','Features were permuted.');
    else
        error('No features or targets in workspace!');
    end
    
case 'HelpPreprocessing'
    h       = findobj(gcbf, 'Tag', 'Preprocessing');
    chosen  = get(h, 'Value');
    evalin('base', ['help ' Preprocessing_methods(chosen).Name ])
    
case 'HelpClassifier'
    h       = findobj(gcbf, 'Tag', 'Algorithm');
    chosen  = get(h, 'Value');
    evalin('base', ['help ' Algorithms(chosen).Name ])
    
case 'About'
    im = imread('about','bmp');
    plot(0,0)
    h  = image(im);
end
