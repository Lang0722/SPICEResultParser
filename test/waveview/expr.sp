* expr probes
.subckt cell a b
r1 a n1 1k
c1 n1 0 1p
r2 n1 b 1k
.ends
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
x1 in out cell
cl out 0 1p
.option post=1 probe post_version=9601
.probe tran par('v(out)*2') vd=v(in,out) dv=deriv('v(out)') x1.loc=par('v(x1.n1)+1')
.probe tran v(out) i(x1.r1) v(in)
.tran 0.5n 10n
.end
