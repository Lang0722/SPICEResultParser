* odd names
.subckt cell a b
r1 a n#1 1k
c1 n#1 0 1p
m1 b n#1 0 0 nch w=1u l=1u
r2 b vdd 10k
vsup vdd 0 1
.ends
.model nch nmos level=1
vin in 0 pulse(0 1 0 1n 1n 5n 10n) ac 1
x1 in out cell
rbus out bus[0] 1k
cbus bus[0] 0 1p
rb2 bus[0] bus<1> 1k
cb2 bus<1> 0 1p
m2 out in 0 0 nch w=1u l=1u
.option post=1 post_version=9601 probe
.probe tran v(out) v(x1.n#1) i(x1.r1) i1(x1.m1) i1(m2) i(rbus) v(bus[0]) v(bus<1>) v(in,out) p(rbus) lx4(m2) i(x1.vsup) v(x1.b)
.probe tran myexpr=par('v(out)*2')
.tran 0.1n 20n
.ac dec 5 1k 1g
.probe ac vdb(out) vp(out) v(out) vr(out) vi(out) vm(out) i(rbus) v(x1.n#1) vdb(x1.n#1)
.end
