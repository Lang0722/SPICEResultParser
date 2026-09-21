RC lowpass
V1 in 0 dc 1 ac 1 pulse(0 1 0 1n 1n 5n 10n)
R1 in out 1k
C1 out 0 1p
.control
set filetype=binary
tran 1n 20n
ac dec 5 1 1G
dc V1 0 1 0.25
write nutmeg_rc_bin.raw tran1.all ac1.all dc1.all
set filetype=ascii
write nutmeg_rc_ascii.raw tran1.all ac1.all dc1.all
quit
.endc
.end
