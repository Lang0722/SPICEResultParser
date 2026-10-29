* deep hierarchy
.subckt leaf a b
r1 a n1 1k
c1 n1 0 1p
r2 n1 b 1k
.ends
.subckt mid a b
xl a m leaf
xl2 m b leaf
.ends
.subckt top a b
xm a t mid
xm2 t b mid
.ends
vin in 0 pulse(0 1 0 1n 1n 5n 10n)
xt in out top
cl out 0 1p
.option post=2
.tran 0.1n 20n
.end
