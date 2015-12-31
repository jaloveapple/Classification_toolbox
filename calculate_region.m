function [region, x, y] = calculate_region(features, region)

%This function recaclulated the plot region

%param = max(max(abs(features)));
%region = [-param param -param param Npoints];

region = [min(features(1,:)) max(features(1,:)) ...
	      min(features(2,:)) max(features(2,:)) region(5)];   
   
if (sign(region(1)) == 1)
   region(1) = region(1) / 1.1;
else
   region(1) = region(1) * 1.1;
end   
if (sign(region(3)) == 1)
   region(3) = region(3) / 1.1;
else
   region(3) = region(3) * 1.1;
end   
if (sign(region(2)) == 1)
   region(2) = region(2) * 1.1;
else
   region(2) = region(2) / 1.1;
end   
if (sign(region(4)) == 1)
   region(4) = region(4) * 1.1;
else
   region(4) = region(4) / 1.1;
end   



%region = [max(abs(features(1,:))) max(abs(features(1,:))) ...
%          max(abs(features(2,:))) max(abs(features(2,:))) region(5)];   


%region(find(region == 0)) = 1;
%region = region.*[-1 1 -1 1 1];

x  	 = linspace (region(1),region(2),region(5));
y		 = linspace (region(3),region(4),region(5));
