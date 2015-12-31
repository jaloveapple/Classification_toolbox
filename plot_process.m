function plot_process(mu)

%Plot the mu's during an algorithm's execution
%Inputs:
%   mu - The location of points
% No outputs

h = findobj('UserData','mu');

if (~isempty(h)),
    delete(h)
end

if (~isempty(mu))
    h = plot(mu(1,:),mu(2,:),'ro');
    set(h,'UserData','mu');
end
    
drawnow
    