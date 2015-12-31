function [features, targets, params, region] = click_points(region) 

%Manually enter points into the workspace
ax = region(1:4);

h   = findobj(findobj('Tag','Main'),'Tag','txtNumberPointsPerClick');
if ~isempty(h),
    N   = str2num(get(h,'String'));
else
    N   = 20;
end

hold on
axis(ax)
mousefrm;
features    = [];
targets     = [];
m0          = [];
m1          = [];
s0          = [];
s1          = [];

distx			= (region(2) - region(1))/20;
disty			= (region(4) - region(3))/20;

%Print the text on figure:
t1 = text(region(1)+distx,region(3)+disty*3,'Class A  => Left  Mouse Click/Drag');
t2 = text(region(1)+distx,region(3)+disty*2,'Class B  => Right Mouse Click/Drag');
t3 = text(region(1)+distx,region(3)+disty,'To Stop  => Click any key on keyboard');

%Coordinates of figure for display:
h = findobj(gcbf,'Tag','Main');
p = get(h,'Position');
wx1 = p(1);
wx2 = p(3);
wy1 = p(2);
wy2 = p(4);


d = 0.000001;

while 1
    %Take the x,y of features per one click 
    %k           = waitforbuttonpress;
    %point1      = get(h,'CurrentPoint');    % button down detected
    [x, y, button] = ginput(1);
    point1      = [x y];
    finalRect   = rbbox;                   % return figure units
    point2      = get(gca,'CurrentPoint');    % button up detected
    if (isempty(point1)),
        point1 = point2;
    end
    point1      = point1(1,1:2);              % extract x and y
    point2      = point2(1,1:2);
  
    mx   = (point1(1) + point2(1))/2;
    my   = (point1(2) + point2(2))/2;
    m    = [mx ; my];

    sx   = abs(point1(1)-point2(1))/5;
    sy   = abs(point1(2)-point2(2))/5;
   
    %Set minimal std to be 0.02
    smin = 0.02;
   
    if sx < smin, sx = smin; end
    if sy < smin, sy = smin; end
  
    sigma = [sx 0 ; 0 sy]; 
   
    %Calculate the points:
    points = sigma * randn(2,N) + m * ones(1,N);
   
    %Check the mouse output in order to set the class or exit :
    switch button
    case 1
        plot(points(1,:),points(2,:),'gx')
        features  = [features,points];
        targets   = [targets,ones(1,N)];
        in        = size(m1,1);
        m1(in+1,:)= m';
        s1(in+1,:,:) = sigma;
      
    case 3
        plot(points(1,:),points(2,:),'bo')
        features  = [features,points];
        targets   = [targets,zeros(1,N)];
        in        = size(m0,1);
        m0(in+1,:)= m';
        s0(in+1,:,:) = sigma;
      
    otherwise
        break
    end
   
    axis(ax)
end

in        = size(m0,1);
w0        = ones(in,1)/in;
in        = size(m1,1);
w1        = ones(in,1)/in;
p0        = length(w0)/(length(w0)+length(w1));

region = [ax,100];

set(t1,'Visible','off')
set(t2,'Visible','off')
set(t3,'Visible','off')
hold off;

params.m0 = m0;
params.m1 = m1;
params.s0 = s0;
params.s1 = s1;
params.w0 = w0;
params.w1 = w1;
params.p0 = p0;
