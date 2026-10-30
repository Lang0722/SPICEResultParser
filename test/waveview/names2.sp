* hierarchical named probes, expressions with odd names, unnamed par
.param rval=1k
.subckt leaf a b
r1 a n1 1k
c1 n1 0 1p
r2 n1 b 1k
.ends
.subckt pair a b
x2 a b leaf
.ends
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
xt in out pair
cl out 0 1p
.option post=1 probe
.probe tran top=v(out) xt.vo=v(xt.x2.n1) xt.pw=p(xt.x2.r2) xt.io=i(xt.x2.r1) xt.x2.deep=v(xt.x2.n1)
+ my$x=par('v(out)*2') q[0]=par('v(out)-1') dv$=deriv('v(out)') par('rval') par('temper') par('v(in)*3')
.tran 1n 5n
.end
