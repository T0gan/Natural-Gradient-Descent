x= -10:0.1:10;
y = -sin(x)./x.^3 ;


plot(x,y);
ylim([-40 40]);
xlim([-6 6]);
xlabel('theta')
ylabel('The Cost Function J(theta)')