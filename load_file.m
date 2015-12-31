function [features, targets, distribution_parameters] = load_file(filename, region)

%Load a file with either data points and/or distribution parameters

features = []; targets = []; distribution_parameters = [];

N		= region(5);
   
if isempty(findstr(filename,'.mat')),
   filename = [filename '.mat'];
end

if (~isempty(dir(filename)))
   load (filename)
   if ((isempty(features) | isempty(targets))) & isempty(distribution_parameters) & ~exist('m0')
      error('No features, targets, or distribution parameters found in this file')
   end
      
   hm = findobj('Tag', 'Messages');
   st= '';
   if (~isempty(features)),
      st = ['File loaded. Found ' num2str(size(targets,2)) ' data points'];
      
      if (size(features,1) > 2),
         %More than two dimensions in the data
         h = feature_selection;
         waitfor(h, 'UserData',1)
         
         h1		 = findobj(h, 'Tag', 'txtHiddenMethod');
         chosen = get(h1, 'String');
         h1		 = findobj(h, 'Tag', 'txtHiddenParams');
         params = get(h1, 'String');
         
         if (~isempty(str2num(params))),
            params = str2num(params);
         end
         
         [features, targets] = feval(chosen, features, targets, params, []);
         
         close(h)
      end
      
      param = max(max(abs(features)));
      plot_scatter(features, targets) 
      axis([-param,param,-param,param])
   end
   
   if ~isempty(distribution_parameters)
	   n0 = size(distribution_parameters.s0,1);
		n1 = size(distribution_parameters.s1,1);
      if (~isempty(st)),
         st = [st ', '];
      else
         st = 'Found ';
      end
      st = [st num2str(n0) ' Gaussians for class 0 and ' num2str(n1) ' Gaussians for class 1.'];         
   end
   
   if exist('m0')
       warning('Located an old-styled distribution file. Please use the new format!')
       distribution_parameters.m0 = m0;
       distribution_parameters.m1 = m1;
       distribution_parameters.w0 = w0;
       distribution_parameters.w1 = w1;
       distribution_parameters.s0 = s0;
       distribution_parameters.s1 = s1;
       distribution_parameters.p0 = p0;
	   n0 = size(distribution_parameters.s0,1);
       n1 = size(distribution_parameters.s1,1);
       if (~isempty(st)),
          st = [st ', '];
       else
          st = 'Found ';
       end
       st = [st num2str(n0) ' Gaussians for class 0 and ' num2str(n1) ' Gaussians for class 1.'];         
   end
   
   set(hm,'String',st)  
else
   hm = findobj('Tag', 'Messages');
   set(hm,'String','File not found.')
end 

if (nargout == 1),
   features = distribution_parameters;
end
