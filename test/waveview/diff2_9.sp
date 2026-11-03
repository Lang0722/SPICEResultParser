* hierarchical differential probes, collision with a top-level node
.subckt leaf a b
r1 a n1 1k
c1 n1 0 1p
r2 n1 b 1k
c2 b 0 1p
.ends
.subckt mid a b
xl a m leaf
r3 m b 1k
.ends
vin in 0 pulse(0 1 0 1n 1n 5n 10n) ac 1
x1 in out mid
rb out b 1k
cb b 0 2p
.option post=1 post_version=9601 probe
.probe tran v(x1.xl.n1,x1.xl.b) v(x1.xl.n1,b) v(x1.xl.n1,x1.m) v(x1.m,x1.xl.n1) v(b)
.probe ac v(x1.xl.n1,x1.xl.b) vm(x1.xl.n1,x1.xl.b) v(x1.xl.n1,b)
.tran 1n 10n
.ac dec 2 1k 1meg
.end
