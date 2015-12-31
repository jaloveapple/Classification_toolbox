function plot_scatter(plot_features, plot_targets, fig, color)

% Make a scatter plot of the data
% Inputs:
%	plot_features	- Data features
%	plot_targets	- Data targets
%	fig				- Optional figure handle
%	color				- Optional color tag (if 1, will color all points red)

switch nargin,
case 2
   color = 0;
case 3
   color = 0;
   figure(fig);
case 4
   if ~isempty(fig),
      figure(fig)
   end
end

one   = find(plot_targets == 1);
zero  = find(plot_targets == 0);

switch color
case 0
   plot(plot_features(1,zero),plot_features(2,zero),'bo',plot_features(1,one),plot_features(2,one),'gx')
case 1
   plot(plot_features(1,zero),plot_features(2,zero),'ro',plot_features(1,one),plot_features(2,one),'rx','LineWidth',2)
case 2
   plot(plot_features(1,zero),plot_features(2,zero),'ko',plot_features(1,one),plot_features(2,one),'mx')   
end

grid on
