* two param sweep 9601
.subckt stage a b rr=1k
r1 a mid rr
c1 mid 0 1n
r2 mid b 10
.ends
.param rval=1k cval=1n
vin in 0 pulse(0 1 0 1n 1n 5u 10u) ac 1
x1 in out stage rr=rval
c2 out 0 cval
.option post=1 post_version=9601
.tran 10n 10u sweep data=pts
.data pts rval cval
+ 1k 1n
+ 2k 1n
+ 2k 2n
.enddata
.meas tran tdel trig v(in) val=0.5 rise=1 targ v(out) val=0.5 rise=1
.meas tran never when v(out)=5 rise=1
.end
