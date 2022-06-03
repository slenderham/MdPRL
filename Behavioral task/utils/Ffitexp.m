function f = Ffitexp(x,xdata)

idle_t = x(1);
steady_state = x(2);
start_state = x(3);
timescale = x(4);

f = steady_state+start_state*exp(-xdata/timescale);
f = [repmat(start_state,[1,idle_t]) f];
f = f(1:length(xdata));