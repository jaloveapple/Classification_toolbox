function [N, classes] = find_classes(targets)

%Find the number of target classes in the targets of a data set

classes = unique(targets);
N       = length(classes);
