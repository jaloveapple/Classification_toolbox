function fig = classifier()
% This is the machine-generated representation of a Handle Graphics object
% and its children.  Note that handle values may change when these objects
% are re-created. This may cause problems with any callbacks written to
% depend on the value of the handle at the time the object was saved.
% This problem is solved by saving the output as a FIG-file.
%
% To reopen this object, just type the name of the M-file at the MATLAB
% prompt. The M-file and its associated MAT-file must be on your path.
% 
% NOTE: certain newer features in MATLAB may not have been saved in this
% M-file due to limitations of this format, which has been superseded by
% FIG-files.  Figures which have been annotated using the plot editor tools
% are incompatible with the M-file/MAT-file format, and should be saved as
% FIG-files.

load classifier

h0 = figure('Units','characters', ...
	'Color',[0.8 0.8 0.8], ...
	'Colormap',mat0, ...
	'CreateFcn','                                                                        ', ...
	'FileName','D:\Users\elad\HW\Classification_toolbox\classifier.m', ...
	'MenuBar','none', ...
	'PaperPosition',[18 180 576 432], ...
	'PaperType','A4', ...
	'Position',[44 11.46 139 45.38], ...
	'Renderer','zbuffer', ...
	'RendererMode','manual', ...
	'Tag','Main', ...
	'ToolBar','none');
h1 = uimenu('Parent',h0, ...
	'Callback','                                    ', ...
	'Label','&文件', ...
	'Tag','&File1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''SearchForFile'')', ...
	'Label','&载入', ...
	'Tag','&FileLoad');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''Save'')', ...
	'Label','保存', ...
	'Tag','&FileSave');
h2 = uimenu('Parent',h1, ...
	'Label','清除', ...
	'Tag','&FileClear1');
h3 = uimenu('Parent',h2, ...
	'Callback','classifier_commands(''ClearBounds'')', ...
	'Label','清除边界', ...
	'Tag','&File&ClearBounds');
h3 = uimenu('Parent',h2, ...
	'Callback','classifier_commands(''ClearWorkspace'')', ...
	'Label','清除工作空间', ...
	'Tag','&File&ClearWorkspace');
% h2 = uimenu('Parent',h1, ...
% 	'Tag','&FileMenu1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''MixFeatures'')', ...
	'Label','&改变特征', ...
	'Tag','&MixFeatures');
% h2 = uimenu('Parent',h1, ...
% 	'Tag','&FileMenu1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''Exit'')', ...
	'Label','&退出', ...
	'Tag','&File&Exit');
h1 = uimenu('Parent',h0, ...
	'Callback','                                    ', ...
	'Label','&编辑', ...
	'Tag','&Edit1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''Print'')        ', ...
	'Label','&打印', ...
	'Tag','&Edit&Print1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''Copy'')         ', ...
	'Label','&复制', ...
	'Tag','&Edit&Copy1');
h1 = uimenu('Parent',h0, ...
	'Callback','                                    ', ...
	'Label','&视图', ...
	'Tag','&View1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''MenuGrid'')     ', ...
	'Checked','on', ...
	'Label','&网格', ...
	'Tag','&View&Grid1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''MenuZoom'')     ', ...
	'Label','&缩放', ...
	'Tag','&View&Zoom1');
h1 = uimenu('Parent',h0, ...
	'Label','&功能', ...
	'Tag','&Actions');
h2 = uimenu('Parent',h1, ...
   'Label','&开始分类', ...
	'Callback','classifier_commands(''Start'')', ...   
	'Tag','&Actions&Start');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''Params'') ', ...
   'Label','查看参数', ...
	'Tag','&Actions&ShowParams');
h2 = uimenu('Parent',h1, ...
   'Label','&查找最优参数', ...
	'Callback','FindParameters', ...   
	'Tag','&Actions&FindBest');
h2 = uimenu('Parent',h1, ...
   'Label','&算法比较', ...
	'Callback','multialgorithms', ...   
	'Tag','&Actions&Start');
h2 = uimenu('Parent',h1, ...
	'Tag','&FileMenu1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''EnterManualDist'')', ...
   'Label','&手动分配', ...
	'Tag','&Actions&ManualDistributions');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''GenerateDistData'')', ...
   'Label','&生成数据集', ...
	'Tag','&Actions&Generatedataset');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''ManualGraphData'')', ...
   'Label','数据集图表', ...
	'Tag','&Actions&Graphicaldataset');
h1 = uimenu('Parent',h0, ...
	'Label','&选项', ...
	'Tag','&Options');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''SeparatePreprocessing'') ', ...
	'Label','&预处理', ...
	'Tag','&Options&SeparatePreprocessing');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''ShowPartitions'') ', ...
	'Label','&在训练过程中显示分类', ...
	'Tag','&Options&ShowPartitions');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''ShowTrainingPoints'') ', ...
	'Label','显示训练集', ...
	'Tag','&Options&ShowTraining');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''ShadeDecisionRegion'') ', ...
	'Label','决策区域', ...
	'Checked','on', ...
	'Tag','&Options&ShadeDecision');
h1 = uimenu('Parent',h0, ...
	'Label','&帮助', ...
	'Tag','&Help');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''HelpPreprocessing'') ', ...
	'Label','预处理算法帮助', ...
	'Tag','&Help&HelpPreprocessing');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''HelpClassifier'') ', ...
	'Label','分类算法帮助', ...
	'Tag','&Help&HelpClassifier');
% h2 = uimenu('Parent',h1, ...
% 	'Tag','&Help1');
h2 = uimenu('Parent',h1, ...
	'Callback','classifier_commands(''About'') ', ...
	'Label','&关于', ...
	'Tag','&About');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[4 1 131 4], ...
	'Style','frame', ...
	'Tag','frmMessages');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[60.5 5.5 74.5 8], ...
	'Style','frame', ...
	'Tag','frmErrors');
h1 = axes('Parent',h0, ...
	'Box','on', ...
	'CameraUpVector',[0 1 0], ...
	'CameraUpVectorMode','manual', ...
	'Color',[1 1 1], ...
	'Position',[0.4353846153846154 0.366852886405959 0.5353846153846154 0.5884543761638734], ...
	'Tag','axsPlotArea', ...
	'XColor',[0 0 0], ...
	'XGrid','on', ...
	'YColor',[0 0 0], ...
	'YGrid','on', ...
	'ZColor',[0 0 0], ...
	'ZGrid','on');
h2 = line('Parent',h1, ...
	'Color',[0 0 1], ...
	'Tag','Line1', ...
	'XData',0, ...
	'YData',0);
h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[-0.005813953488372103 -1.151898734177215 17.32050807568877], ...
	'Tag','Text4', ...
	'VerticalAlignment','cap');
set(get(h2,'Parent'),'XLabel',h2);
h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[-1.191860465116279 -0.006329113924050667 17.32050807568877], ...
	'Rotation',90, ...
	'Tag','Text3', ...
	'VerticalAlignment','baseline');
set(get(h2,'Parent'),'YLabel',h2);
h2 = text('Parent',h1, ...
	'Color',[0 0 0], ...
	'HandleVisibility','off', ...
	'HorizontalAlignment','center', ...
	'Position',[-0.005813953488372103 1.044303797468354 17.32050807568877], ...
	'Tag','Text1', ...
	'VerticalAlignment','bottom');
set(get(h2,'Parent'),'Title',h2);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[4 40 49 3], ...
	'Style','frame', ...
	'Tag','frmFilename');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[4 22.5 49 17], ...
	'Style','frame', ...
	'Tag','frmParameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 41 30 1], ...
	'String','文件名: ', ...
	'Style','text', ...
	'Tag','StaticText1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'Callback','classifier_commands(''FileNameInput'')', ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[16 40.5 30 1.75], ...
	'Style','edit', ...
	'Tag','FileNameInput');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''SearchForFile'')', ...
	'ListboxTop',0, ...
	'Position',[47 40.5 5 1.75], ...
	'String','...', ...
	'Tag','pshLocateFile');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'Callback','                 ', ...
	'ListboxTop',0, ...
	'Position',[44 35.25 8 1.5], ...
	'String','1', ...
	'Style','edit', ...
	'Tag','Redraws');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 35.25 25 1.5], ...
	'String','重复次数: ', ...
	'Style','text', ...
	'Tag','StaticText1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 33.25 35 1.5], ...
	'String','训练集比例: ', ...
	'Style','text', ...
	'Tag','StaticText1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'Callback','                           ', ...
	'ListboxTop',0, ...
	'Position',[44 33.25 8 1.5], ...
	'String','20', ...
	'Style','edit', ...
	'Tag','PercentTraining');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 31.25 35 1.5], ...
	'String','预处理: ', ...
	'Style','text', ...
	'Tag','StaticText2');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''Changed Preprocessing'')', ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[20 31.25 30 1.5], ...
	'String',mat3, ...
	'Style','popupmenu', ...
	'Tag','Preprocessing', ...
	'Value',1);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'ListboxTop',0, ...
	'Position',[41 29.25 11 1.5], ...
	'Style','edit', ...
	'Tag','txtPreprocessingParameters', ...
	'Visible','off');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 29.25 35 1.5], ...
	'Style','text', ...
	'Tag','lblPreprocessingParameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 27.25 35 1.5], ...
	'String','算法: ', ...
	'Style','text', ...
	'Tag','StaticText1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''Changed Algorithm'')', ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[20 27.25 32 1.5], ...
	'String',mat4, ...
	'Style','popupmenu', ...
	'Tag','Algorithm', ...
	'Value',5);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'ListboxTop',0, ...
	'Position',[44 25.25 8 1.5], ...
	'Style','edit', ...
	'Tag','txtAlgorithmParameters', ...
	'Visible','off');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'ListboxTop',0, ...
	'Position',[5 23.25 47 1.5], ...
	'Style','edit', ...
	'Tag','txtAlgorithmParametersLong', ...
	'Visible','off');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 25 37 1.5], ...
	'Style','text', ...
	'Tag','lblAlgorithmParameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 37.25 32 1.5], ...
	'String','评价方法:', ...
	'Style','text', ...
	'Tag','txtRedrawingmethod1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'ListboxTop',0, ...
	'Position',[32 37.25 20 1.5], ...
	'String',mat5, ...
	'Style','popupmenu', ...
	'Tag','popErrorEstimationMethod', ...
	'Value',1);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[62 9.5 60 1], ...
	'Style','text', ...
	'Tag','TestSetError');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[62 8 60 1], ...
	'Style','text', ...
	'Tag','TrainSetError');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[62 6.5 60 1], ...
	'Style','text', ...
	'Tag','BayesError');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'FontSize',10, ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[6 1.5 125 1.5], ...
	'Style','text', ...
	'Tag','Messages');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'FontSize',10, ...
	'FontWeight','bold', ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[62 11 60 2], ...
	'String','分类错误率:', ...
	'Style','text', ...
	'Tag','txtErrors');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'FontSize',10, ...
	'FontWeight','bold', ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[6 3.5 80 1.25], ...
	'String','消息:', ...
	'Style','text', ...
	'Tag','txtMessages');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[4 10 49 12], ...
	'Style','frame', ...
	'Tag','frmDistInput');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''EnterManualDist'')', ...
	'HorizontalAlignment','center', ...
	'ListboxTop',0, ...
	'Position',[5 19.5 47 1.75], ...
	'String','手动配置', ...
	'Tag','pshManual');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 12.75 30 1.5], ...
	'Style','text', ...
    'String','每次点击选择点的数量:', ...
	'Tag','lblNumberofManualPoints');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'Callback','                           ', ...
	'ListboxTop',0, ...
	'Position',[45 12.75 7 1.5], ...
	'String','20', ...
	'Style','edit', ...
	'Tag','txtNumberPointsPerClick');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'ListboxTop',0, ...
	'Position',[5 10.75 40 1.5], ...
	'String','显示贝叶斯决策边界', ...
	'Style','checkbox', ...
	'Tag','chkBayes', ...
	'Value',1);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''GenerateDistData'')', ...
	'HorizontalAlignment','center', ...
	'ListboxTop',0, ...
	'Position',[5 17.25 47 1.75], ...
	'String','产生样本数据集', ...
	'Tag','pshGenerateData');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''ManualGraphData'')', ...
	'HorizontalAlignment','center', ...
	'ListboxTop',0, ...
	'Position',[5 15 47 1.75], ...
	'String','图表展示', ...
	'Tag','pshGraphData');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','multialgorithms', ...
	'ListboxTop',0, ...
	'Position',[4 6 10 3], ...
	'String','比较', ...
	'Tag','MultiAlgorithms');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''Params'')', ...
	'ListboxTop',0, ...
	'Position',[17 6 10 3], ...
	'String','参数', ...
	'Tag','Parameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''ClearBounds'')', ...
	'ListboxTop',0, ...
	'Position',[30 6 10 3], ...
	'String','清除', ...
	'Tag','Clear');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...
	'Callback','classifier_commands(''Start'')', ...   
	'FontWeight','bold', ...
	'ListboxTop',0, ...
	'Position',[43 6 10 3], ...
	'String','开始', ...
	'Tag','Start');
if nargout > 0, fig = h0; end
classifier_commands('Init');
