* ac only real columns 9601
vin in 0 ac 1
r1 in out 1k
c1 out 0 1n
.option post=1 post_version=9601 probe
.probe ac vm(out) vp(out) vdb(out)
.ac dec 2 1k 1meg
.end
