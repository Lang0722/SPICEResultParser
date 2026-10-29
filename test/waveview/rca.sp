* rc sweep ascii
.param rval=1k cval=1n
vin in 0 pulse(0 1 0 1n 1n 5u 10u) ac 1
r1 in out rval
c1 out 0 cval
.option post=2
.tran 10n 20u sweep rval poi 3 1k 2k 4k
.ac dec 10 1k 100meg sweep cval poi 2 1n 2n
.dc vin 0 1 0.1 sweep rval poi 2 1k 3k
.meas tran tdel trig v(in) val=0.5 rise=1 targ v(out) val=0.5 rise=1
.meas tran vmax max v(out)
.meas tran never when v(out)=5 rise=1
.meas ac f3db when vdb(out)=-3 fall=1
.meas dc vo5 find v(out) when v(in)=0.5
.end
