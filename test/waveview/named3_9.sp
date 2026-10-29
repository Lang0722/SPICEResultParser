* named probes of other kinds
.model nch nmos level=1
vin in 0 pulse(0 1 0 1n 1n 5n 10n) ac 1
r1 in out 1k
c1 out 0 1p
m1 out in 0 0 nch w=1u l=1u
.option post=1 post_version=9601 probe
.probe tran pw=p(r1) gmx=lx7(m1) idm=i1(m1) vdiff=v(in,out)
.probe ac vodb=vdb(out) vo=v(out) io=i(r1)
.tran 1n 10n
.ac dec 2 1k 1meg
.end
